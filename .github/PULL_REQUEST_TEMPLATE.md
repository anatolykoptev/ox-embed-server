## Summary

<!-- What changed and why -->

## Checklist

- [ ] `make ci` passes locally
- [ ] No `unwrap`/`expect`/`panic` in non-test, non-startup code
- [ ] No silent errors on writes (DB/HTTP/file failures log or bump a metric)
- [ ] No magic numbers without const + doc
