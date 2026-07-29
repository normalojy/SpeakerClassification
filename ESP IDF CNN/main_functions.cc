#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model_data.h"

#include "driver/uart.h"

#include "esp_timer.h"

extern "C" {
    #include "ssd1306.h"
}

#define UART_PORT UART_NUM_0   // usually UART0 is mapped to the USB port
#define BUF_SIZE 1024
#include <random>

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
// int inference_count = 0;

constexpr int kTensorArenaSize = 25000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(tensorflowlite_cnn_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<15> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (resolver.AddRelu() != kTfLiteOk) {
    return;
  }
  if (resolver.AddAdd() != kTfLiteOk) {
    return;
  }
  if (resolver.AddMul() != kTfLiteOk) {
    return;
  }
  if (resolver.AddLogistic() != kTfLiteOk) {
    return;
  }
  if (resolver.AddReshape() != kTfLiteOk) {
    return;
  }
  if (resolver.AddQuantize() != kTfLiteOk) {
    return;
  }
  if (resolver.AddDequantize() != kTfLiteOk) {
    return;
  }
  if (resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (resolver.AddShape() != kTfLiteOk) {
    return;
  }
  if (resolver.AddStridedSlice() != kTfLiteOk) {
    return;
  }
  if (resolver.AddPack() != kTfLiteOk) {
    return;
  }
  if (resolver.AddConv2D() != kTfLiteOk) {
    return;
  }
  if (resolver.AddMaxPool2D() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Important check: Does the model really have 2 inputs?
  if (input->dims->size != 1 || input->dims->data[0] != 2) {
    MicroPrintf("Warning: Expected input shape [2], got different shape\n");
  }

  MicroPrintf("Model ready. Input shape: [%d], Output shape: [%d]\n",
              input->dims->data[0], output->dims->data[0]);

MicroPrintf("\n=== Model Input/Output Information ===\n");

// Print Input Tensor Shape
MicroPrintf("Input tensor shape: [");
for (int i = 0; i < input->dims->size; i++) {
    MicroPrintf("%d", input->dims->data[i]);
    if (i < input->dims->size - 1) {
        MicroPrintf(", ");
    }
}
MicroPrintf("]  (rank = %d)\n", input->dims->size);

// Print Output Tensor Shape
MicroPrintf("Output tensor shape: [");
for (int i = 0; i < output->dims->size; i++) {
    MicroPrintf("%d", output->dims->data[i]);
    if (i < output->dims->size - 1) {
        MicroPrintf(", ");
    }
}
MicroPrintf("]\n");

  // Configure UART parameters
  uart_config_t uart_config = {
    .baud_rate = 115200,
    .data_bits = UART_DATA_8_BITS,
    .parity    = UART_PARITY_DISABLE,
    .stop_bits = UART_STOP_BITS_1,
    .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
  };
  uart_param_config(UART_PORT, &uart_config);

  // Install UART driver with RX buffer
  uart_driver_install(UART_PORT, BUF_SIZE, 0, 0, NULL, 0);

  init_ssd1306();
}

void loop() {

  char buffer[32]; 

  size_t bytes_available = 0;
  uart_get_buffered_data_len(UART_NUM_0, &bytes_available);

    //   for (int i = 0; i < 1024; i++) {
    //     uint8_t pixel = 0;
    //     input->data.f[i] = pixel / 255.0f;      // 0-255 Int -> 0-1 Float
    //   }

    // // Run inference, and report any error
    // int64_t before = esp_timer_get_time();
    // TfLiteStatus invoke_status = interpreter->Invoke();
    // int64_t after = esp_timer_get_time();
    // if (invoke_status != kTfLiteOk) {
    //   MicroPrintf("Invoke failed\n");
    //   return;
    // }


    // // Find predicted digit (assuming 4 classes)
    // float max_prob = -1e9f;
    // int predicted_speaker = 0;

    // float confidence = 0.0f;

    // if (output->type == kTfLiteFloat32) {
    //   for (int i = 0; i < 4; i++) {
    //     if (output->data.f[i] > max_prob) {
    //       max_prob = output->data.f[i];
    //       predicted_speaker = i;
    //     }
    //   }
    // } 

    // confidence = max_prob;

    // int64_t end = esp_timer_get_time();

    // // Format the prediction into a string
    // ssd1306_clear();
    // snprintf(buffer, sizeof(buffer), "Speaker: %d", predicted_speaker + 1);
    // ssd1306_print_str(0, 0, buffer, false);

    // snprintf(buffer, sizeof(buffer), "Conf: %.1f%%", confidence * 100);
    // ssd1306_print_str(0, 16, buffer, false);

    // int64_t diff_us = after - before;              // microseconds
    // double diff_s = (double)diff_us / 1e6;         // convert to seconds

    // snprintf(buffer, sizeof(buffer), "Time: %.3f s", diff_s);
    // // snprintf(buffer, sizeof(buffer), "Time: %lli", after - before);
    // ssd1306_print_str(0, 32, buffer, false);

    // ssd1306_display();

  if (bytes_available >= 1024) {               // 32 * 32 = 1024
    int64_t start = esp_timer_get_time();
    if (input->type == kTfLiteFloat32) {
      for (int i = 0; i < 1024; i++) {
        uint8_t pixel = 0;
        uart_read_bytes(UART_NUM_0, &pixel, 1, 0);
        input->data.f[i] = pixel / 255.0f;      // 0-255 Int -> 0-1 Float
      }
    }

    // Run inference, and report any error
    int64_t before = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter->Invoke();
    int64_t after = esp_timer_get_time();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed\n");
      return;
    }


    // Find predicted digit (assuming 4 classes)
    float max_prob = -1e9f;
    int predicted_speaker = 0;

    float confidence = 0.0f;

    if (output->type == kTfLiteFloat32) {
      for (int i = 0; i < 4; i++) {
        if (output->data.f[i] > max_prob) {
          max_prob = output->data.f[i];
          predicted_speaker = i;
        }
      }
    } 

    confidence = max_prob;

    int64_t end = esp_timer_get_time();

    // Format the prediction into a string
    ssd1306_clear();
    snprintf(buffer, sizeof(buffer), "Speaker: %d", predicted_speaker + 1);
    ssd1306_print_str(0, 0, buffer, false);

    snprintf(buffer, sizeof(buffer), "Conf: %.1f%%", confidence * 100);
    ssd1306_print_str(0, 16, buffer, false);

    int64_t diff_us = after - before;              // microseconds
    double diff_s = (double)diff_us / 1e6;         // convert to seconds

    snprintf(buffer, sizeof(buffer), "Time: %.3f s", diff_s);

    // snprintf(buffer, sizeof(buffer), "Time: %lli", after - before);
    ssd1306_print_str(0, 32, buffer, false);

    // snprintf(buffer, sizeof(buffer), "End: %lli", end - start);
    // ssd1306_print_str(0, 48, buffer, false);

    ssd1306_display();
  }
  
  // // memset(input->data.f, 255, 32*32*sizeof(float));

  // // Run inference, and report any error
  // TfLiteStatus invoke_status = interpreter->Invoke();
  // if (invoke_status != kTfLiteOk) {
  //   MicroPrintf("Invoke failed\n");
  //   return;
  // }


  // // Find predicted digit (assuming 10 classes)
  // float max_prob = -1e9f;
  // int predicted_speaker = 0;

  // float confidence = 0.0f;

  // if (output->type == kTfLiteFloat32) {
  //   for (int i = 0; i < 4; i++) {
  //     if (output->data.f[i] > max_prob) {
  //       max_prob = output->data.f[i];
  //       predicted_speaker = i;
  //     }
  //   }
  // } 

  // confidence = max_prob;

  // MicroPrintf("Predicted Speaker: %d   Confidence: %.1f%%\n", predicted_speaker, confidence * 100);
}