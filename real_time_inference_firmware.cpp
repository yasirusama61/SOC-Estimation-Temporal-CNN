#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "FreeRTOS.h"
#include "task.h"
#include "timers.h"
#include "semphr.h"

// Model data generated with TensorFlow Lite conversion
extern const uint8_t model_data[];  // Model data array generated from the TFLite model

// TensorFlow Lite settings
constexpr int kTensorArenaSize = 10 * 1024;  // 1024 Hz processor for MCU
uint8_t tensor_arena[kTensorArenaSize];

// Define a debugging flag for conditional compilation
#define DEBUG_MODE 1

// Mutex for serial logging
SemaphoreHandle_t xSerialSemaphore;

// Function prototypes
void initialize_inference();
void capture_sensor_data(float* input_data);
float run_inference();
void log_inference_result(float inference_result, unsigned long inference_time);

// Task declarations
void Task_Inference(void* pvParameters);

// TFLite interpreter, model, and tensor pointers (global for reuse in functions)
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Initialization function for the TensorFlow Lite interpreter
void initialize_inference() {
    static tflite::AllOpsResolver resolver;
    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        #ifdef DEBUG_MODE
        Serial.println("Error: Model schema version does not match!");
        #endif
        vTaskDelete(NULL);  // Stop execution if there's an error
    }

    interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter->AllocateTensors();

    // Assign input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
}

// Function to simulate sensor data acquisition (replace with actual hardware sensor code)
void capture_sensor_data(float* input_data) {
    for (int i = 0; i < input->bytes / sizeof(float); i++) {
        input_data[i] = analogRead(A0) * (3.3 / 1023.0);  // reading from ADC (adjust as necessary)
    }
}

// Run the inference and measure time
float run_inference() {
    unsigned long start_time = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long end_time = micros();

    if (invoke_status != kTfLiteOk) {
        #ifdef DEBUG_MODE
        Serial.println("Error: Inference failed!");
        #endif
        return -1;
    }

    unsigned long inference_time = end_time - start_time;
    float inference_result = output->data.f[0];  // Assuming single output
    
    #ifdef DEBUG_MODE
    log_inference_result(inference_result, inference_time);
    #endif

    return inference_result;
}

// Function to log the inference result and timing
void log_inference_result(float inference_result, unsigned long inference_time) {
    if (xSemaphoreTake(xSerialSemaphore, (TickType_t)10) == pdTRUE) {
        Serial.print("Inference Result: ");
        Serial.println(inference_result);
        Serial.print("Inference Time (us): ");
        Serial.println(inference_time);
        xSemaphoreGive(xSerialSemaphore);
    }
}

// FreeRTOS task to handle periodic inference
void Task_Inference(void* pvParameters) {
    while (true) {
        // Capture sensor data
        capture_sensor_data(input->data.f);

        // Run inference
        float result = run_inference();

        // Process result (e.g., trigger actions, send data) if needed
        if (result >= 0) {
            // Example action based on inference result
            if (result > 0.5) {
                // Take some action
            }
        }

        // Delay to control task frequency (adjust based on application requirements)
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// FreeRTOS setup function
void setup() {
    Serial.begin(115200);

    // Create a mutex for serial access (to avoid concurrent logging issues)
    xSerialSemaphore = xSemaphoreCreateMutex();

    // Initialize the TensorFlow Lite interpreter
    initialize_inference();

    // Create the Inference Task with a higher priority to ensure it runs as a real-time task
    xTaskCreate(Task_Inference, "InferenceTask", 2048, NULL, 2, NULL);
}

// Main loop remains empty as FreeRTOS handles task scheduling
void loop() {
    // Intentionally left empty as FreeRTOS manages the task lifecycle
}
