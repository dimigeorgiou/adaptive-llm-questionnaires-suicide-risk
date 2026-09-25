# V2 example inputs (synthetic)

Entirely synthetic, non-clinical example files for running V2 without Google or
OpenAI credentials. They contain no patient data. The categories are generic
well-being domains chosen for the demo. They are not a validated instrument and
deliberately contain no suicide-screening items (those belong to validated
instruments administered under the service's protocol).

```bash
export ADAPTIVE_QUESTIONNAIRES_CONFIG=config/config.example.ini
python main.py -o v2_select --session examples/v2/session_S001-s1.json \
  --candidates examples/v2/candidates_S001-s1.json --state-dir ./v2_state
# fill the Likert columns of v2_state/S001/S001-s1_proposal.csv, then:
python main.py -o v2_import_feedback --subject S001 --session S001-s1 \
  --feedback v2_state/S001/S001-s1_proposal.csv --state-dir ./v2_state
python main.py -o v2_select --session examples/v2/session_S001-s2.json \
  --candidates examples/v2/candidates_S001-s2.json --state-dir ./v2_state
```
