"""Logique de la page Molière (fenetreEnvoyee, apresEchec) exécutée par node.

Les fonctions pures sont extraites de page.html entre deux marqueurs : ce sont celles que le navigateur
exécute vraiment. Le résultat est aussi validé par le vrai contrôle du serveur (demo.server.checked_messages)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import demo.server as demo

PAGE = Path(__file__).resolve().parents[1] / 'app_moliere' / 'page.html'
pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='node absent')


def logique():
    html = PAGE.read_text(encoding='utf-8')
    m = re.search(r'/\* LOGIQUE-PURE-DEBUT.*?\*/(.*?)/\* LOGIQUE-PURE-FIN \*/', html, re.S)
    assert m, 'marqueurs de la logique pure introuvables dans page.html'
    return m.group(1)


def node(code):
    res = subprocess.run(['node', '-e', logique() + '\n' + code], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def U(t):
    return {'role': 'user', 'content': t}


def A(t):
    return {'role': 'assistant', 'content': t}


def fenetre(messages, limites=None):
    return node(f'console.log(JSON.stringify(fenetreEnvoyee({json.dumps(messages)}, {json.dumps(limites)})));')


def valide_serveur(suite):
    demo.checked_messages(suite)   # lève ValueError si le serveur refuserait la suite


def test_historique_normal_inchange():
    msgs = [U('a'), A('b'), U('c')]
    assert fenetre(msgs, None) == msgs


def test_deux_messages_utilisateur_de_suite():
    """Défaut 1 : [U1, A1, U2 resté sans réponse, U3] ne donne plus 400 « doivent alterner »."""
    suite = fenetre([U('u1'), A('a1'), U('u2'), U('u3')])
    assert suite == [U('u1'), A('a1'), U('u3')]
    valide_serveur(suite)


def test_apres_echec_puis_nouvel_envoi():
    """Le scénario du relecteur de bout en bout : U1→A1, U2 en erreur, U3 envoyé."""
    res = node(f"""
      let m = [{json.dumps(U('u1'))}, {json.dumps(A('a1'))}, {json.dumps(U('u2'))}];
      const r = apresEchec(m, 'Le serveur local ne répond plus.');
      m = r.messages;
      const afficheEtRetire = JSON.stringify(m);
      m = m.filter((x) => !x.erreur); m.push({{role: 'user', content: 'u3'}});
      console.log(JSON.stringify({{reprise: r.reprise, apres: JSON.parse(afficheEtRetire), envoye: fenetreEnvoyee(m, null)}}));
    """)
    assert res['reprise'] == 'u2'
    assert [x['role'] for x in res['apres']] == ['user', 'assistant', 'assistant']
    assert res['apres'][-1]['erreur'] is True and res['apres'][-1]['reprise'] == 'u2'
    assert res['envoye'] == [U('u1'), A('a1'), U('u3')]
    valide_serveur(res['envoye'])


def test_apres_echec_premier_message():
    res = node("console.log(JSON.stringify(apresEchec([{role:'user',content:'salut'}], 'oups')));")
    assert res['reprise'] == 'salut' and len(res['messages']) == 1 and res['messages'][0]['erreur']


def test_plus_de_19_messages():
    msgs = []
    for i in range(14):
        msgs += [U(f'u{i}'), A(f'a{i}')]
    msgs.append(U('dernier'))                      # 29 messages
    suite = fenetre(msgs, {'messages': 19})
    assert len(suite) <= 19 and suite[0]['role'] == 'user' and suite[-1] == U('dernier')
    assert suite[0] == U('u5')                     # les plus vieux échanges partent deux par deux
    valide_serveur(suite)


def test_depassement_des_12000_caracteres():
    msgs = []
    for i in range(6):
        msgs += [U(f'{i}' * 2000), A(f'{i}' * 2000)]
    msgs.append(U('fin'))                          # 24 003 caractères sur 13 messages
    suite = fenetre(msgs, {'caracteres_total': 12000})
    assert sum(len(m['content']) for m in suite) <= 12000
    assert suite[-1] == U('fin') and suite[0]['role'] == 'user' and len(suite) % 2 == 1
    valide_serveur(suite)


def test_message_unique_trop_long():
    suite = fenetre([U('x' * 5000)])
    assert len(suite) == 1 and len(suite[0]['content']) == 2000
    valide_serveur(suite)


def test_historique_bancal_est_redresse():
    suite = fenetre([A('orphelin'), U('u1'), {'role': 'assistant', 'content': 'oups', 'erreur': True},
                     A('a1'), A('a1bis'), U('u2'), A('reste en fin')])
    assert suite == [U('u1'), A('a1bis'), U('u2')]
    assert fenetre([A('seul')]) == []


def test_proprietes_sur_des_historiques_aleatoires():
    """500 historiques tirés au hasard (graine fixe) : la suite envoyée est toujours acceptée par le serveur."""
    suites = node("""
      let graine = 12345;
      const alea = (n) => { graine = (graine * 1103515245 + 12345) % 2147483648; return graine % n; };
      const sorties = [];
      for (let k = 0; k < 500; k++) {
        const n = alea(45); const m = [];
        for (let i = 0; i < n; i++) {
          const t = alea(4);
          m.push({ role: alea(3) === 0 ? 'assistant' : 'user', content: 'x'.repeat(1 + alea(t === 0 ? 3500 : 400)),
                   erreur: alea(12) === 0 });
        }
        sorties.push(fenetreEnvoyee(m, { messages: 19, caracteres_message: 2000, caracteres_total: 12000 }));
      }
      console.log(JSON.stringify(sorties));
    """)
    vides = 0
    for suite in suites:
        if not suite:
            vides += 1
            continue
        valide_serveur(suite)
    assert vides < 100


def test_la_page_utilise_ces_fonctions():
    html = PAGE.read_text(encoding='utf-8')
    assert 'fenetreEnvoyee(messages, etat && etat.limites)' in html and 'apresEchec(messages' in html
    assert 'AbortController' in html and '120000' in html


def test_etiquette_du_personnage_memorisee_au_debut_de_la_demande():
    """Défaut 6 : le nom est lu avant l'appel réseau, et la liste est bloquée pendant le calcul."""
    html = PAGE.read_text(encoding='utf-8')
    corps = html[html.index('async function demander()'):html.index('async function envoyer(')]
    assert corps.index('const nom = nomPersonnage()') < corps.index("await api('/api/chat'")
    assert 'persona: nom }' in corps and 'persona: nomPersonnage()' not in corps
    assert "$('personnage').disabled = enCours" in html
