# Adjacent-layer CPU expert transition prefetch: rejected

The opt-in path learned adjacent-layer route transitions and used a separate worker to fill one
predicted next-layer expert into the normal-RAM cache. It preserved the argmax and forced token
stream, but did not beat the stable 7.54 tok/s plain-generation reference:

| arm | N | throughput | useful / submitted | CPU RAM misses |
| --- | ---: | ---: | ---: | ---: |
| control | 64 | 6.90 tok/s | 0 / 0 | 7,608 |
| bounded temporal prediction | 64 | 7.13 tok/s | 62 / 128 | 7,605 |
| ranked non-HBM prediction | 64 | 7.01 tok/s | 1,448 / 4,895 | 7,588 |

The apparent single-run gain came without a material miss reduction, and aggressive prediction
spent thousands of fills for only 20 fewer misses while remaining below the stable baseline. The
implementation, loader ABI extension, telemetry, and flags were removed. Raw logs are
`hy3-rungen-transition-*-clean-forced64-warm128-run1.log`.
