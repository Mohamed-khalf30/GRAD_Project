#ifndef PREPROCESSING_H
#define PREPROCESSING_H

// Model input shape: [1, 40, 7]
// Features: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, delta_t
const int   NUM_FEATURES = 7;
const int   WINDOW_SIZE  = 40;
const float THRESHOLD    = 0.99f;

const float SCALER_MEAN[] = {
  -0.235942f,   // acc_x
  -0.024933f,   // acc_y
  10.286900f,   // acc_z
   0.010179f,   // gyro_x
  -0.106352f,   // gyro_y
  -0.027440f    // gyro_z
};

const float SCALER_SCALE[] = {
  5.883090f,    // acc_x
  5.335825f,    // acc_y
  4.114338f,    // acc_z
  1.047432f,    // gyro_x
  0.509599f,    // gyro_y
  0.821975f     // gyro_z
};

// ✅ 20Hz = 0.05s per sample
const float MAX_DELTA_T = 0.05f;

// ✅ Butterworth order 4, fs=20Hz, cutoff=5Hz
// scipy: butter(4, 5.0/(20.0/2), btype='low')
const float FILTER_B[] = {
  0.0939752562f,
  0.3759010247f,
  0.5638515371f,
  0.3759010247f,
  0.0939752562f
};

const float FILTER_A[] = {
  1.0000000000f,
  0.0000000000f,
  0.4860942669f,
  0.0000000000f,
  0.0176648539f
};

const int FILTER_ORDER = 4;

#endif // PREPROCESSING_H