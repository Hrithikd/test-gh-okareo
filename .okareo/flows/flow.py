import os
import sys
import json
import tempfile
from okareo import Okareo
from okareo.model_under_test import CustomModel, ModelInvocation
from okareo.reporter import JSONReporter
from okareo_api_client.models.test_run_type import TestRunType

# Your API key
OKAREO_API_KEY = os.getenv("OKAREO_API_KEY", "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjYyNDNmYWY3In0.eyJzdWIiOiIzN2NjMmZkYi04YmNjLTRmYzYtYTE5NC1lYjY2YzgyMDc0NTQiLCJ0eXBlIjoidGVuYW50QWNjZXNzVG9rZW4iLCJ0ZW5hbnRJZCI6IjM5YTlmZmUzLTEzZjYtNDQ2MS05ZTZkLWMzNzIyOGIwY2QwNyIsImFwcGxpY2F0aW9uSWQiOiJmY2Q3ZGQzNy0yZjgxLTQ5YTAtOGI0My1kZTQxZTZjODA5Y2YiLCJyb2xlcyI6WyJGRVRDSC1ST0xFUy1CWS1BUEkiXSwicGVybWlzc2lvbnMiOlsiRkVUQ0gtUEVSTUlTU0lPTlMtQlktQVBJIl0sImF1ZCI6IjYyNDNmYWY3LTk2YzYtNDgxNS1iOTMzLWNlZGY5ZGY1Yjg5MCIsImlzcyI6Imh0dHBzOi8vb2thcmVvLnVzLmZyb250ZWdnLmNvbSIsImlhdCI6MTc2NTkzMzcyNX0.MUlXThZY0JEcvmNHBixHbWIDuEOnzcMlbacQtb7YLvbqT-jxFXg88I4d-NyggzzZZX_XAcpgxNmK7yTrYReEWVHiLh0SrLBpVE67Lnr13C1iOnFUOmsRVnsjxKEJsdFzqVNHVf8ubxqqqgv8C3ijTltYrgysIoYFf0O67A1XAQeI9hX8TJVEDVTkWBzCJ6JXZM3wNMymaVy7nVcvs9J9PQ7aAtLqmiuc-ye6ztfi--pa_Kn_1JYUPPAO2G7lBfO8Ryu-urcV9doKHo5baLGzZRV9lVVnBPC9SjZTXMcgDcZYTOKDNNRq8Dt3EHdvG5barDAuAeSP1bB36d6swPW9ug")

os.environ["OKAREO_API_KEY"] = OKAREO_API_KEY

# Define minimum thresholds for each metric
METRICS_MIN = {
    "coherence": 3.0,
    "consistency": 3.0,
    "fluency": 3.0,
}

def check_metrics_pass(eval_run, metrics_min):
    """Check if evaluation metrics meet minimum thresholds"""
    if not eval_run.model_metrics or not eval_run.model_metrics.mean_scores:
        print("❌ No metrics found in evaluation results")
        return False
    
    mean_scores = eval_run.model_metrics.mean_scores.to_dict()
    passed = True
    
    print("\n📊 Evaluation Results:")
    print("-" * 50)
    
    for metric, min_threshold in metrics_min.items():
        if metric in mean_scores:
            score = mean_scores[metric]
            status = "✅ PASS" if score >= min_threshold else "❌ FAIL"
            print(f"{metric:15} {score:.2f} / {min_threshold:.2f} {status}")
            if score < min_threshold:
                passed = False
        else:
            print(f"{metric:15} Not found in results")
    
    print("-" * 50)
    return passed

try:
    # Initialize Okareo client
    okareo = Okareo(OKAREO_API_KEY)
    
    # Create a simple scenario (test data)
    scenario_data = [
        {"input": "sunny day", "result": "What a beautiful sunny day!"},
        {"input": "rainy weather", "result": "The rainy weather is refreshing."},
        {"input": "snowy mountain", "result": "The snowy mountain looks majestic."},
    ]
    
    # Upload the scenario
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in scenario_data:
            f.write(json.dumps(item) + '\n')
        temp_file = f.name
    
    scenario = okareo.upload_scenario_set(
        file_path=temp_file,
        scenario_name="Simple Generation Test Scenario"
    )
    
    # Clean up temp file
    os.remove(temp_file)
    
    # Define a simple custom model
    class SimpleGenerationModel(CustomModel):
        """A very simple custom model that generates text based on input."""
        
        def invoke(self, input_value):
            # Simple logic: create a sentence from the input
            generated_text = f"This is a generated response about: {input_value}"
            
            return ModelInvocation(
                model_prediction=generated_text,
                model_input=input_value,
                model_output_metadata={"input": input_value, "output": generated_text}
            )
    
    # Register the model
    model_under_test = okareo.register_model(
        name="Simple Custom Generation Model",
        model=SimpleGenerationModel(name="simple_gen_model")
    )
    
    # Run evaluation
    print("🚀 Running evaluation...")
    evaluation = model_under_test.run_test(
        name="Simple Generation Evaluation",
        scenario=scenario,
        test_run_type=TestRunType.NL_GENERATION,
        calculate_metrics=True,
        checks=['coherence_summary', 'consistency_summary', 'fluency_summary']
    )
    
    # Use JSONReporter to log results
    reporter = JSONReporter([evaluation])
    
    print("\n" + "="*50)
    print("📝 Evaluation Report")
    print("="*50)
    
    # Log the full report
    reporter.log()
    
    # Check if metrics pass thresholds
    passed = check_metrics_pass(evaluation, METRICS_MIN)
    
    # Print summary
    print(f"\n🔗 View detailed results: {evaluation.app_link}")
    print(f"📋 Test Run ID: {evaluation.id}")
    
    if passed:
        print("\n✅ Evaluation PASSED - All metrics meet minimum thresholds")
        sys.exit(0)
    else:
        print("\n❌ Evaluation FAILED - Some metrics below minimum thresholds")
        sys.exit(1)
        
except Exception as error:
    print(f"\n❌ CI failed because of an error calling Okareo: {error}")
    sys.exit(1)