package ai.phaseone.runtime.portable;

import com.dylibso.chicory.runtime.HostFunction;
import com.dylibso.chicory.runtime.ImportValues;
import com.dylibso.chicory.runtime.Instance;
import com.dylibso.chicory.runtime.InterpreterMachine;
import com.dylibso.chicory.wasm.Parser;
import com.dylibso.chicory.wasm.WasmModule;
import com.dylibso.chicory.wasm.types.ExternalType;
import com.dylibso.chicory.wasm.types.FunctionType;
import com.dylibso.chicory.wasm.types.ValType;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Experimental, interpreted control programs over an already-installed native host.
 *
 * Tensors stay in the host. Programs receive opaque request/result handles, not
 * addresses or Android objects. No WASI, network, filesystem, or native loader
 * imports are provided. The expected digest MUST come from a verified catalog.
 * This class supplies integrity checking, not publisher authentication.
 */
public final class PortableAdapter {
    public static final int HOST_ABI = 1;
    public static final int MAX_MODULE_BYTES = 64 * 1024;
    private static final String NAMESPACE = "dtr_native_v1";
    private final WasmModule module;
    private final String digest;
    private final int instructionLimit;
    private final int nativeCallLimit;

    /** DTR's native binding owns all handles, validation, and inference resources.
     * Implementations must be scoped to one invocation; they are not exposed to
     * arbitrary Java reflection or code loading by the adapter.
     */
    public interface NativeHost {
        boolean isAvailable(int engine);
        int infer(int engine, int requestHandle);
    }

    public static final class Fault extends RuntimeException {
        public final String code;
        public Fault(String code, String message) { super(message); this.code = code; }
        public Fault(String code, String message, Throwable cause) { super(message, cause); this.code = code; }
    }

    public static final class Result {
        public final int outputHandle;
        public final int instructions;
        public final int nativeCalls;
        public final String adapterSha256;
        private Result(int handle, int instructions, int calls, String digest) {
            this.outputHandle = handle;
            this.instructions = instructions;
            this.nativeCalls = calls;
            this.adapterSha256 = digest;
        }
    }

    public PortableAdapter(byte[] wasm, String expectedSha256) {
        this(wasm, expectedSha256, 10_000, 8);
    }

    public PortableAdapter(byte[] wasm, String expectedSha256, int instructionLimit, int nativeCallLimit) {
        Objects.requireNonNull(wasm);
        if (wasm.length < 8 || wasm.length > MAX_MODULE_BYTES)
            throw new Fault("invalid_adapter", "Adapter size is outside its limit");
        if (instructionLimit < 1 || instructionLimit > 1_000_000 || nativeCallLimit < 1 || nativeCallLimit > 64)
            throw new IllegalArgumentException("Adapter execution limits are outside their bounds");
        // Copy before hashing: a caller cannot race mutation between verification and parsing.
        byte[] verified = wasm.clone();
        digest = sha256(verified);
        if (expectedSha256 == null || !expectedSha256.matches("[0-9a-f]{64}") || !digest.equals(expectedSha256))
            throw new Fault("integrity_failed", "Adapter differs from its verified catalog digest");
        if (verified[0] != 0 || verified[1] != 'a' || verified[2] != 's' || verified[3] != 'm')
            throw new Fault("invalid_adapter", "Expected WebAssembly bytecode");
        this.instructionLimit = instructionLimit;
        this.nativeCallLimit = nativeCallLimit;
        try { module = Parser.parse(verified); }
        catch (RuntimeException e) { throw new Fault("invalid_adapter", "Invalid WebAssembly module", e); }
        // ABI 1 is a small control program over host-owned buffers. Linear memory,
        // tables, and start functions are deliberately outside this initial ABI.
        if (module.memorySection().isPresent() || module.tableSection().tableCount() != 0 || module.startSection().isPresent())
            throw new Fault("unsupported_adapter", "ABI 1 supports handle-based control programs only");
        for (int i = 0; i < module.importSection().importCount(); i++) {
            var imported = module.importSection().getImport(i);
            if (imported.importType() != ExternalType.FUNCTION || !NAMESPACE.equals(imported.module())
                    || !("available".equals(imported.name()) || "infer".equals(imported.name())))
                throw new Fault("unsupported_host_abi", "Adapter requests an unavailable host import");
        }
    }

    /** Invoke on the application's inference executor, never on its UI thread.
     * A fresh interpreter instance isolates each run. Native calls are synchronous:
     * the instruction budget does not interrupt an in-flight vendor inference.
     */
    public Result execute(NativeHost host, int requestHandle) {
        Objects.requireNonNull(host);
        if (requestHandle <= 0) throw new Fault("invalid_handle", "Request handles must be positive");
        int[] instructions = {0};
        int[] calls = {0};
        Set<Integer> producedHandles = new HashSet<>();
        HostFunction available = new HostFunction(NAMESPACE, "available",
                FunctionType.of(List.of(ValType.I32), List.of(ValType.I32)),
                (instance, args) -> new long[]{host.isAvailable((int) args[0]) ? 1 : 0});
        HostFunction infer = new HostFunction(NAMESPACE, "infer",
                FunctionType.of(List.of(ValType.I32, ValType.I32), List.of(ValType.I32)),
                (instance, args) -> {
                    if (++calls[0] > nativeCallLimit) throw new Fault("native_call_limit", "Adapter exceeded its native call budget");
                    int engine = (int) args[0];
                    int input = (int) args[1];
                    if (input != requestHandle && !producedHandles.contains(input))
                        throw new Fault("invalid_handle", "Adapter used a handle outside this invocation");
                    if (!host.isAvailable(engine)) throw new Fault("engine_unavailable", "Adapter requested an unregistered native engine");
                    int output = host.infer(engine, input);
                    if (output <= 0 || output == requestHandle || producedHandles.contains(output))
                        throw new Fault("invalid_handle", "Native host must return a fresh result handle");
                    producedHandles.add(output);
                    return new long[]{output};
                });
        try {
            Instance instance = Instance.builder(module)
                    .withStart(false)
                    .withMachineFactory(InterpreterMachine::new)
                    .withImportValues(ImportValues.builder().addFunction(available, infer).build())
                    .withUnsafeExecutionListener((instruction, stack) -> {
                        if (++instructions[0] > instructionLimit)
                            throw new Fault("instruction_limit", "Adapter exceeded its instruction budget");
                    }).build();
            if (!FunctionType.of(List.of(ValType.I32), List.of(ValType.I32)).equals(instance.exportType("run")))
                throw new Fault("invalid_adapter", "Adapter run export must accept and return one i32 handle");
            long[] output = instance.export("run").apply(requestHandle);
            if (output.length != 1 || !producedHandles.contains((int) output[0]))
                throw new Fault("invalid_handle", "Adapter must return a result produced by native inference");
            return new Result((int) output[0], instructions[0], calls[0], digest);
        } catch (Fault e) { throw e; }
        catch (RuntimeException e) { throw new Fault("adapter_execution_failed", "Portable adapter failed", e); }
    }

    public String sha256() { return digest; }

    public static String sha256(byte[] value) {
        try {
            byte[] bytes = MessageDigest.getInstance("SHA-256").digest(value);
            StringBuilder result = new StringBuilder(64);
            for (byte b : bytes) result.append(Character.forDigit((b >>> 4) & 15, 16)).append(Character.forDigit(b & 15, 16));
            return result.toString();
        } catch (NoSuchAlgorithmException e) { throw new IllegalStateException(e); }
    }
}
