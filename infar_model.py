import onnxruntime as ort
import numpy as np
import os

# المسار المطلق للنموذج
model_path = os.path.abspath("models/titanic_model.onnx")
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
def predict(input_data):
    
    # Ensure input_data is a numpy array
    
    input_data = np.array(input_data, dtype=np.float32)

    # Run inference
    result = session.run([output_name], {input_name: input_data})[0]
    
    
    probabilities =float(result[0][0])
    prediction=int(probabilities > 0.5)
 

    
    # Return the prediction result
    return prediction


if __name__ == "__main__":
    # Example usage
    example_input = np.array([[3, 22, 1, 0, 7.25,0,1]], dtype=np.float32)  # Example input data
    prediction = predict(example_input)
    print(f"Prediction: {prediction}")  # Output the prediction result