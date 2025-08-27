/* Copyright 2022 The TensorFlow Authors. All Rights Reserved. */
#include <Arduino.h>
#include "micro_features_micro_features_generator.h"

#include <cmath>
#include <cstring>

#include "micro_features_micro_model_settings.h"
#include "tensorflow/lite/micro/micro_log.h"

// Configure FFT to output 16 bit fixed point.
#define FIXED_POINT 16

namespace {

bool g_is_first_time = true;

}  // namespace


TfLiteStatus GenerateMicroFeatures(const int16_t* input, int input_size,
                                   int output_size, int8_t* output) {

  // Temporary float buffer for mel output
  float mel_output[kFeatureSliceSize];
  ComputeLogMelSlice(input, mel_output); //fills up mel_output  

  float scale = 0.0825823f;
  int zero_point = 39;
  for (size_t i = 0; i < kFeatureSliceSize ; ++i) {
    // the outputs of mel_output are float, normal 32bit float
    // that's the whole point, you know only have to de-quanatize the way you want to baby
    // you only have to copy the quanatized mel_output[i] into output[i]

    int32_t quantized = roundf(mel_output[i] / scale) + zero_point;
    quantized = std::min<int32_t>(127, std::max<int32_t>(-128, quantized));
    output[i] = static_cast<int8_t>(quantized);  // Or any quantization scheme

  }

  return kTfLiteOk;
}
