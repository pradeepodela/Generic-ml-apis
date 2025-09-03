from transformers import pipeline
from gliclass import GLiClassModel, ZeroShotClassificationPipeline

# Define email test cases with expected labels
test_cases = [
    {
        "text": "Can’t remember my password—please reset it.",
        "expected": ["Support"]
    },
    {
        "text": "What are your business hours tomorrow?",
        "expected": ["Enquiry"]
    },
    {
        "text": "I’d like a quote for 50 licenses of your Pro plan.",
        "expected": ["Sales"]
    },
    {
        "text": "Great webinar today—thanks for hosting!",
        "expected": ["Other"]
    },
    {
        "text": "Ticket #48219 was marked as solved, but the export still fails with ‘500 Internal Server Error’. I’ve attached the HAR file and a screencast.",
        "expected": ["Support"]
    },
    {
        "text": "We’re comparing analytics vendors for a July rollout. Can your product meet GDPR-compliant data residency in the EU? A feature sheet would help.",
        "expected": ["Enquiry"]
    },
    {
        "text": "Following our call, here’s the signed NDA. Next step: please send the custom pricing for 5M monthly events plus onboarding services.",
        "expected": ["Sales"]
    },
    {
        "text": "FYI: your latest release notes list ‘June 31’—might want to fix the date.",
        "expected": ["Other"]
    },
    {
        "text": "Since updating to v3.2 our integrations tab is blank. We rely on the Salesforce sync for daily ops. Restarting the connector didn’t help (see attached logs). Could you escalate this? We’re prepping for quarter-end tomorrow.",
        "expected": ["Support"]
    },
    {
        "text": "Hello, I’m evaluating AI transcription tools for a non-profit research project. Budget is limited, but accuracy is key (medical terminology). Does your academic discount apply, and can you share any peer-reviewed benchmarks?",
        "expected": ["Enquiry"]
    },
    {
        "text": "Hi Chris, Loved the demo! If you can do 15 % off annual pre‑pay and include priority support SLA, our CFO is ready to sign this week. Let me know so we can align on contract language.",
        "expected": ["Sales"]
    },
    {
        "text": "Team—thanks for the rapid hotfix last night. Everything’s stable now; our board noticed the fast turnaround.",
        "expected": ["Other"]
    },
]

# Labels to classify
classes_verbalized = ["Support", "Enquiry", "Sales", "Other"]
hypothesis_template = "This email is about {}"

# Load the classifier
zeroshot_classifier = pipeline(
    "zero-shot-classification",
    model="knowledgator/gliclass-large-v1.0-init",
)

# Evaluation
correct = 0
total = len(test_cases)

print("\n--- Classification Results ---\n")
for i, case in enumerate(test_cases):
    result = zeroshot_classifier(
        case["text"],
        classes_verbalized,
        hypothesis_template=hypothesis_template,
        multi_label=True
    )
    predicted = [label for label, score in zip(result['labels'], result['scores']) if score > 0.5]
    expected = case["expected"]
    
    match = set(expected).issubset(set(predicted))
    
    print(f"Test {i+1}:")
    print(f"Text: {case['text']}")
    print(f"Predicted: {predicted}")
    print(f"Expected: {expected}")
    print(f"Match: {'✅' if match else '❌'}\n")
    
    if match:
        correct += 1

# Summary
print("--- Summary ---")
print(f"Total Test Cases: {total}")
print(f"Correct Predictions: {correct}")
print(f"Accuracy: {correct/total:.2%}")
