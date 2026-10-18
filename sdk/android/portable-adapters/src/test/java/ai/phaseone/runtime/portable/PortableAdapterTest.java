package ai.phaseone.runtime.portable;

import com.dylibso.chicory.wabt.Wat2Wasm;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import static org.junit.Assert.*;

public class PortableAdapterTest {
    private static String route(int engine) {
        return "(module (import \"dtr_native_v1\" \"infer\" (func $infer (param i32 i32) (result i32))) "
                + "(func (export \"run\") (param i32) (result i32) i32.const " + engine + " local.get 0 call $infer))";
    }
    private static PortableAdapter adapter(String wat) {
        byte[] wasm = Wat2Wasm.parse(wat);
        return new PortableAdapter(wasm, PortableAdapter.sha256(wasm));
    }
    private static final class Host implements PortableAdapter.NativeHost {
        final List<Integer> engines = new ArrayList<>();
        public boolean isAvailable(int engine) { return engine == 1 || engine == 2; }
        public int infer(int engine, int input) { engines.add(engine); return 100 + engines.size(); }
    }
    @Test public void newModuleChangesNativeCallsAndExistingModuleStillWorks() {
        var original = adapter(route(1));
        var updated = adapter(route(2));
        var host = new Host();
        assertEquals(101, original.execute(host, 1).outputHandle);
        assertEquals(102, updated.execute(host, 1).outputHandle);
        assertEquals(103, original.execute(host, 1).outputHandle);
        assertEquals(List.of(1, 2, 1), host.engines);
        assertNotEquals(original.sha256(), updated.sha256());
    }
    @Test public void integrityFailureNeverInvokesHost() {
        byte[] wasm = Wat2Wasm.parse(route(1));
        assertEquals("integrity_failed", assertThrows(PortableAdapter.Fault.class,
                () -> new PortableAdapter(wasm, "0".repeat(64))).code);
    }
    @Test public void unknownHostAbiRejectedBeforeExecution() {
        assertEquals("unsupported_host_abi", assertThrows(PortableAdapter.Fault.class,
                () -> adapter(route(1).replace("dtr_native_v1", "dtr_native_v2"))).code);
    }
    @Test public void unavailableEngineNeverInvoked() {
        var host = new Host();
        assertEquals("engine_unavailable", assertThrows(PortableAdapter.Fault.class,
                () -> adapter(route(99)).execute(host, 1)).code);
        assertTrue(host.engines.isEmpty());
    }
    @Test public void endlessProgramIsBounded() {
        String wat = "(module (func (export \"run\") (param i32) (result i32) (loop $again br $again) i32.const 1))";
        assertEquals("instruction_limit", assertThrows(PortableAdapter.Fault.class,
                () -> adapter(wat).execute(new Host(), 1)).code);
    }
    @Test public void nativeInvocationBudgetIsEnforced() {
        String wat = "(module (import \"dtr_native_v1\" \"infer\" (func $infer (param i32 i32) (result i32))) "
                + "(func (export \"run\") (param i32) (result i32) (loop $again i32.const 1 local.get 0 call $infer drop br $again) i32.const 1))";
        var host = new Host();
        assertEquals("native_call_limit", assertThrows(PortableAdapter.Fault.class,
                () -> adapter(wat).execute(host, 1)).code);
        assertEquals(8, host.engines.size());
    }
    @Test public void fabricatedResultHandleRejected() {
        assertEquals("invalid_handle", assertThrows(PortableAdapter.Fault.class,
                () -> adapter("(module (func (export \"run\") (param i32) (result i32) i32.const 1))").execute(new Host(), 1)).code);
    }
    @Test public void fabricatedRequestHandleRejectedBeforeNativeCall() {
        var host = new Host();
        assertEquals("invalid_handle", assertThrows(PortableAdapter.Fault.class,
                () -> adapter(route(1).replace("local.get 0", "i32.const 42")).execute(host, 1)).code);
        assertTrue(host.engines.isEmpty());
    }
    @Test public void moduleMemoryIsOutsideInitialAbi() {
        assertEquals("unsupported_adapter", assertThrows(PortableAdapter.Fault.class,
                () -> adapter("(module (memory 1) (func (export \"run\") (param i32) (result i32) i32.const 1))")).code);
    }
    @Test public void implicitStartExportDoesNotRun() {
        String wat = "(module (import \"dtr_native_v1\" \"infer\" (func $infer (param i32 i32) (result i32))) "
                + "(func (export \"_start\") i32.const 2 i32.const 1 call $infer drop) "
                + "(func (export \"run\") (param i32) (result i32) i32.const 1 local.get 0 call $infer))";
        var host = new Host();
        adapter(wat).execute(host, 1);
        assertEquals(List.of(1), host.engines);
    }
    @Test public void exportSignatureMustMatchHostAbi() {
        var host = new Host();
        assertEquals("invalid_adapter", assertThrows(PortableAdapter.Fault.class,
                () -> adapter("(module (func (export \"run\") (param i32) (result i64) i64.const 101))").execute(host, 1)).code);
        assertTrue(host.engines.isEmpty());
    }
}
