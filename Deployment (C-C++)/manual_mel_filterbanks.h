#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// main function of mel-filterbanks energies.
bool ComputeLogMelSlice(const int16_t* input, float* output);

// /// @brief Load a test input slice into internal buffer (for testing only)
// void LoadTestSlice();

#ifdef __cplusplus
}
#endif
