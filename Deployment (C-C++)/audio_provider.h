/* Copyright 2022 The TensorFlow Authors. All Rights Reserved. */

#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_AUDIO_PROVIDER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_AUDIO_PROVIDER_H_

#include "tensorflow/lite/c/common.h"

// function get's the audio that was recorded at ~`start_ms` (could be one 1ms wrong
// by my calculations), it grabs `duration_ms` length amount of samples, 
// it should fill the pointer `audio_samples` with a pointer to the audio extracted
// and fill the pointer `audio_samples_size` with the number of samples extracted.
TfLiteStatus GetAudioSamples(int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples);

// this function should return the last time that the recorded got new audio data.
int32_t LatestAudioTimestamp();

// Starts audio capture.
TfLiteStatus InitAudioRecording();

#endif  // TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_AUDIO_PROVIDER_H_
