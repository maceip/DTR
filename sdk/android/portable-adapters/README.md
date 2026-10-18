# Embedded portable adapters · experimental

Update a small execution program while keeping inference in the app's installed
native engines. Customers install only the application. This Android module uses
Chicory's interpreter; it does not create a WebView or another Android package.
The base Android runtime remains usable without this experimental module.

The initial host interface is deliberately narrow: `dtr_native_v1.available`
checks a registered engine, and `dtr_native_v1.infer` invokes it with an opaque
request handle. Model data and tensors stay in the native host. The downloaded
program returns a result handle. This version supports execution control, not
arbitrary vendor bindings, tokenizer implementations, or downloaded native code.

```java
// These bytes and their digest come from a publisher-authenticated release.
var adapter = new PortableAdapter(verifiedWasm, catalogSha256);
var result = adapter.execute(nativeBindings, requestHandle);
```

The bindings are DTR's internal backend extension point. Application developers
should continue using the application-level inference API. Automatic integration
with `ReleaseManager`, catalog publication, and app lifecycle scheduling remains
to be built. Hash verification here does not replace catalog authentication.

Each invocation gets a fresh interpreter instance, an instruction budget, a
native-call budget, and scoped handles. No filesystem, networking, reflection,
native-library loading, linear memory, or table imports are exposed. Execute on
the app's inference executor. The budget cannot interrupt a vendor call already
in flight, and this prototype is not a hardened isolation boundary for native bugs.

## Device result

On a Pixel 11 Pro XL running Android 17, one installed APK downloaded successive
adapter modules and ran the same small model through LiteRT 2.3 CPU, ExecuTorch
1.5.1 CPU, and LiteRT 2.3's native OpenCL GPU path. All returned `[5, 5, 5, 5]`.
The APK hash, version, install timestamp, and application process stayed unchanged
during adapter updates. Logs show LiteRT delegated the model's sole sum node to
`LITERT_CL`. A cached adapter also ran after an app restart with the download
server stopped.

Wrong hashes, invalid publisher signatures, a newer host ABI, and a looping
adapter were rejected while the working release remained usable. A retained
original adapter remained callable after a new one was activated.

**All native libraries were in the initial APK.** This proves downloadable control
logic can preserve native execution, including this small GPU graph. It does not
prove NPU support, a performance gain, Play acceptance, or independent delivery of
a new native engine. See the [bounded device receipt](evidence/pixel-embedded-update.json).

## Build and checks

This module targets Android API 33+ and pins Chicory 1.7.5. The existing native SDK
still targets API 31+. From `sdk/android`, with JDK 21 and Android SDK 37:

```sh
./gradlew :portable-adapters:testDebugUnitTest :portable-adapters:assembleDebug
```

Tests exercise version switching, retained modules, digest failures, unavailable
engines, ABI mismatches, scoped handles, instruction/native-call limits, and the
entry-point contract. They use real WASM programs with a test host; the separate
Pixel probe supplied the native engine evidence above.
