# Gate audit

This is the retained first gate attempt, not the final clean verdict.

Every live check completed, but
`crash-pp2-forced-spec-server.log` line 161 records this immediately after
`[server] drain complete in 0.0s; exiting`:

```
[worker] spec pending flush failed (DriverError(CUDA_ERROR_DEINITIALIZED, "<Failure when calling cuGetErrorString()>")); dropping session
```

The harness was tightened to wait for 0.5 seconds of worker-reported idle before
shutdown and to reject this signature. See the later gate receipt for the final
verdict.
