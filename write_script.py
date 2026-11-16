import json
import random

def generate_example(idx):
    categories = ['billing', 'technical', 'account', 'shipping', 'general']
    teams = {'billing': 'payments', 'technical': 'support_tech', 'account': 'customer_success', 'shipping': 'logistics', 'general': 'general_help'}
    ticket_types = {
        'billing': ['charged twice', 'refund request', 'subscription invoice error', 'pricing question'],
        'technical': ['api error 500', 'login failure', 'app crash on startup', 'database timeout'],
        'account': ['password reset', 'delete my account', 'update email address', 'two-factor auth issue'],
        'shipping': ['where is my order', 'damaged package', 'wrong item delivered', 'change shipping address'],
        'general': ['how to use feature', 'feature request', 'partner inquiry', 'feedback']
    }
    cat = random.choice(categories)
    sub = random.choice(ticket_types[cat])
    state = f'Ticket #{1000 + idx}: Customer reports issue regarding {sub} in category {cat}.'
    team_options = {'payments': 'Billing', 'support_tech': 'Tech', 'customer_success': 'Account', 'logistics': 'Shipping', 'general_help': 'General'}
    correct_team = teams[cat]
    target_team = {t: (0.85 if t == correct_team else 0.03) for t in team_options}
    tot = sum(target_team.values())
    target_team = {k: v/tot for k, v in target_team.items()}
    escalate = cat in ['billing', 'technical']
    target_yesno = {'yes': 0.9 if escalate else 0.1, 'no': 0.1 if escalate else 0.9}
    urgency_level = random.randint(0, 4)
    target_score = {str(i): (0.8 if i == urgency_level else 0.05) for i in range(5)}
    tot_s = sum(target_score.values())
    target_score = {k: v/tot_s for k, v in target_score.items()}
    questions = [
        {'id': 'team', 'type': 'choice', 'instructions': 'Route to team.', 'options': team_options, 'target': target_team},
        {'id': 'escalate', 'type': 'yesno', 'instructions': 'Escalate?', 'options': {'yes': 'Yes', 'no': 'No'}, 'target': target_yesno},
        {'id': 'urgency', 'type': 'score', 'instructions': 'Urgency level.', 'options': {str(i): f'L{i}' for i in range(5)}, 'target': target_score}
    ]
    return {'state': state, 'questions': questions}

def main():
    random.seed(42)
    data = [generate_example(i) for i in range(1400)]
    with open('train.jsonl', 'w', encoding='utf-8') as f:
        for x in data[:1000]:
            f.write(json.dumps(x) + '\n')
    with open('calibration.jsonl', 'w', encoding='utf-8') as f:
        for x in data[1000:1200]:
            f.write(json.dumps(x) + '\n')
    with open('test.jsonl', 'w', encoding='utf-8') as f:
        for x in data[1200:]:
            f.write(json.dumps(x) + '\n')
    print('Datasets created successfully.')

if __name__ == '__main__':
    main()
