#ifndef PREPROCESSING_H
#define PREPROCESSING_H

const int NUM_FEATURES = 10;
const int WINDOW_SIZE = 10;
const float THRESHOLD = 0.35;

const float SCALER_MEAN[] = {
  0.406645f,
  -10.856946f,
  9.273352f,
  0.000230f,
  -0.018023f,
  -0.001923f,
  -0.023246f,
  0.025071f,
  -0.002870f
};

const float SCALER_SCALE[] = {
  642.336488f,
  1025.459086f,
  16.104902f,
  0.420106f,
  0.243795f,
  0.388164f,
  0.745091f,
  0.665434f,
  0.031183f
};

const float MAX_DELTA_T = 1.249099f;

const float FILTER_B[] = {
  0.0465829066f,
  0.1863316265f,
  0.2794974398f,
  0.1863316265f,
  0.0465829066f
};

const float FILTER_A[] = {
  1.0000000000f,
  -0.7820951980f,
  0.6799785269f,
  -0.1826756978f,
  0.0301188750f
};

const int FILTER_ORDER = 4;

#endif
