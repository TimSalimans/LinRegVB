% compile Kalman filter/smoother
load PoundDollar
sampleExample = coder.typeof(10,[inf,1]);
codegen KalmanFilterAndSmoother -args {y,y,sampleExample,sampleExample}
codegen FilterLogLike -args {y,y,sampleExample,sampleExample}