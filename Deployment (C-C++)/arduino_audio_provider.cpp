/* Copyright 2022 The TensorFlow Authors. All Rights Reserved. */

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#ifndef ARDUINO_EXCLUDE_CODE

#include <algorithm>
#include <cmath>

#include "PDM.h"
#include "audio_provider.h"
#include "micro_features_micro_model_settings.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "test_over_serial/test_over_serial.h"

using namespace test_over_serial;

namespace {
bool g_is_audio_initialized = false;
// An internal buffer able to fit the (DEFAULT_PDM_BUFFER_SIZE 512) * 16 samples,
// at 16 Khz this means ~0.5s of audio. 
constexpr int kAudioCaptureBufferSize = DEFAULT_PDM_BUFFER_SIZE * 16;
int16_t g_audio_capture_buffer[kAudioCaptureBufferSize];
// A buffer that holds our output
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
// Mark as volatile so we can check in a while loop to see if
// any samples have arrived yet.
volatile int32_t g_latest_audio_timestamp = 0;


// test_over_serial sample index
uint32_t g_test_sample_index;
// test_over_serial silence insertion flag
bool g_test_insert_silence = false;
}  // namespace

void CaptureSamples() {
  // each sample -> int16_t, meaning 2 bytes, capturing just filles the microphone
  // buffer then calls this function. the buffer size in bytes / 2 is the number of 
  // samples since each sample is 2 bytes.
  const int number_of_samples = DEFAULT_PDM_BUFFER_SIZE / 2;

  // Calculate what timestamp the last audio sample represents
  const int32_t time_in_ms =
      g_latest_audio_timestamp +
      (number_of_samples / (kAudioSampleFrequency / 1000)); //"time of last recorded sample"
      // this is due to the microphone promising to record kAudioSampleFrequency per second.

  // Determine the index, in the history of all samples, of the last sample
  const int32_t start_sample_offset =
      g_latest_audio_timestamp * (kAudioSampleFrequency / 1000);
  // Determine the index of this sample in our ring buffer
  const int capture_index = start_sample_offset % kAudioCaptureBufferSize;
  // Read the data to the correct place in our buffer
  int num_read =
      PDM.read(g_audio_capture_buffer + capture_index, DEFAULT_PDM_BUFFER_SIZE);
  if (num_read != DEFAULT_PDM_BUFFER_SIZE) {
    MicroPrintf("### short read (%d/%d) @%dms", num_read,
                DEFAULT_PDM_BUFFER_SIZE, time_in_ms);
    while (true) {
      // NORETURN
    }
  }

  // we don't actually call this method, ever
  // the microphone calls it when it's buffer get's filled
  // with an intterupt, that's why we make the timestamp voltile..
  g_latest_audio_timestamp = time_in_ms; 

}

TfLiteStatus InitAudioRecording() {
  if (!g_is_audio_initialized) {
    // Hook up the callback that will be called with each sample
    PDM.onReceive(CaptureSamples);
    // Start listening for audio: MONO @ 16KHz
    if (!PDM.begin(1, kAudioSampleFrequency)) {
      Serial.println("Error: PDM.begin failed");
      return kTfLiteError;
    }    
    // gain: -20db (min) + 6.5db (13) + 3.2db (builtin) = -10.3db - I'll tune this later
    PDM.setGain(13);
    // Block until we have our first audio sample
    // note: that if this wasn't voltile, the compiler would "optimize" this 
    // so that it never escapes the while loop, which usually would work but
    // PDM on receive interupts the flow of the code from outside, so we need
    // the compiler to constantly check if it changed or not...
    while (!g_latest_audio_timestamp) { 
    }
    g_is_audio_initialized = true;
  }

  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  // This next part should only be called when the main thread notices that the
  // latest audio sample data timestamp has changed, so that there's new data
  // in the capture ring buffer. The ring buffer will eventually wrap around and
  // overwrite the data, but the assumption is that the main thread is checking
  // often enough and the buffer is large enough that this call will be made
  // before that happens.

  // Determine the index, in the history of all samples, of the first
  // sample we want
  const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
  // Determine how many samples we want in total
  const int duration_sample_count =
      duration_ms * (kAudioSampleFrequency / 1000);

  // the assumption here is that the output_buffer is big enough for one slice
  // and that you would only call this method to get one slice of data each time.
  for (int i = 0; i < duration_sample_count; ++i) {
    // For each sample, transform its index in the history of all samples into
    // its index in g_audio_capture_buffer
    const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
    // Write the sample to the output buffer
    g_audio_output_buffer[i] = g_audio_capture_buffer[capture_index];
  }

  // Set pointers to provide access to the audio
  *audio_samples_size = duration_sample_count;
  *audio_samples = g_audio_output_buffer;

  return kTfLiteOk;
}


namespace {

void InsertSilence(const size_t len, int16_t value) {
  for (size_t i = 0; i < len; i++) {
    const size_t index = (g_test_sample_index + i) % kAudioCaptureBufferSize;
    g_audio_capture_buffer[index] = value;
  }
  g_test_sample_index += len;
}

int32_t ProcessTestInput(TestOverSerial& test) {
  constexpr size_t samples_16ms = ((kAudioSampleFrequency / 1000) * 16);

  InputHandler handler = [](const InputBuffer* const input) {
    if (0 == input->offset) {
      // don't insert silence
      g_test_insert_silence = false;
    }

    for (size_t i = 0; i < input->length; i++) {
      const size_t index = (g_test_sample_index + i) % kAudioCaptureBufferSize;
      g_audio_capture_buffer[index] = input->data.int16[i];
    }
    g_test_sample_index += input->length;

    if (input->total == (input->offset + input->length)) {
      // allow silence insertion again
      g_test_insert_silence = true;
    }
    return true;
  };

  test.ProcessInput(&handler);

  if (g_test_insert_silence) {
    // add 16ms of silence just like the PDM interface
    InsertSilence(samples_16ms, 0);
  }

  // Round the timestamp to a multiple of 64ms,
  // This emulates the PDM interface during inference processing.
  g_latest_audio_timestamp = (g_test_sample_index / (samples_16ms * 4)) * 64;
  return g_latest_audio_timestamp;
}

}  // namespace


int32_t LatestAudioTimestamp() {
  // CaptureSamples() updated the timestamp
  return g_latest_audio_timestamp;
  
}

#endif  // ARDUINO_EXCLUDE_CODE
