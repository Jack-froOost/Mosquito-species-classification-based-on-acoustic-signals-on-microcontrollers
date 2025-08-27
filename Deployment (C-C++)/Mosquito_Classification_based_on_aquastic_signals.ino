#include <TensorFlowLite.h>

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "main_functions.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#undef PROFILE_MICRO_SPEECH

// Globals
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

constexpr int kTensorArenaSize = 90 * 1024;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace

void setup() {
  Serial.begin(11520);
  while(!Serial); //delete this when deploying fully
  delay(1000);

  Serial.println("Starting...");
  tflite::InitializeTarget();

  Serial.println("Getting the model...");
  model = tflite::GetModel(g_model); //this doesn't envolve parsing or copying. lightweight.
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  Serial.println("Setting up the operations...");

  // Reserve space for 7 operations
  static tflite::MicroMutableOpResolver<7> micro_op_resolver;
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    Serial.println("Failed to add DepthwiseConv2D...");
    return;
  }
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) {
    Serial.println("Failed to add Conv2D...");
    return;
  }
  if (micro_op_resolver.AddRelu() != kTfLiteOk) {
    Serial.println("Failed to add ReLU...");
    return;
  }
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) {
    Serial.println("Failed to add MaxPool2D...");
    return;
  }
  if (micro_op_resolver.AddMean() != kTfLiteOk) {
    Serial.println("Failed to add Mean...");
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("Failed to add FullyConnected...");
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    Serial.println("Failed to add Softmax...");
    return;
  }



  Serial.println("Building an interpreter...");
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  Serial.println("Allocating tensors...");
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  Serial.println("getting a pointer to the input tensor...");
  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) ||
      (model_input->dims->data[1] != kFeatureSliceCount) ||
      (model_input->dims->data[2] != kFeatureSliceSize) ||
      (model_input->dims->data[3] != 1) || //channels
      (model_input->type != kTfLiteInt8)) {
    Serial.println("BAD model input dimentions...");
    MicroPrintf("Microprint: Bad input tensor parameters in model");
    return;
  }
  
  // pointer to the actual data in the tensorArena, curtasy of TFLM...
  // this means you can just copy the flat featureData array here,
  // without worrying about the 4 dimensions of my implementation 
  Serial.println("Getting a pointer to the flat buffer input...");
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from a microphone
  // that will provide the inputs to the neural network.
  Serial.println("Creating a feature provider...");
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  Serial.println("Creating an output smoother...");
  static RecognizeCommands static_recognizer;
  recognizer = &static_recognizer;

  // this will store the last FeatureData slice time
  previous_time = 0;

  Serial.println("Initlization audio recoroding...");
  // start the audio
  TfLiteStatus init_status = InitAudioRecording();
  if (init_status != kTfLiteOk) {
    MicroPrintf("Unable to initialize audio");
    return;
  }

  Serial.println("Initialization complete... Ready to listen to some audio !");
  // MicroPrintf("Initialization complete... Ready to listen to some audio !");

}

// The name of this function is important for Arduino compatibility.
void loop() {
  
  // last time an audio batch came in
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    MicroPrintf("Feature generation failed");
    return;
  }
  previous_time += how_many_new_slices * kFeatureSliceStrideMs;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }



  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  // Run the model on the mel-filterbanks energies input.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);

  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    MicroPrintf("RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
    // we have the time, label, confidince, and is it a new command.
  // now just make a pretty output, probably on a screen ...
  RespondToCommand(current_time, found_command, score, is_new_command);
  // delay(10000);


}
