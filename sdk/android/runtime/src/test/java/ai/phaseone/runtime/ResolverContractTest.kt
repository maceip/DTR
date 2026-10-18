package ai.phaseone.runtime

import org.junit.Assert.*
import org.junit.Test
import java.io.File

class ResolverContractTest {
    private val fixtures = parseObject(File("../../contracts/resolver-cases.json").readBytes())

    @Test fun sharedPythonAndroidSelectionContract() {
        for (element in fixtures.array("cases")) {
            val case = element.asJsonObject
            val catalog = decodeCatalog(json.toJson(case["catalog"]).toByteArray(), CatalogTrust(emptyMap()), true)
            val profile = json.fromJson(case["device"], DeviceProfile::class.java)
            val policy = case.child("policy")
            val request = ReleasePolicy(policy.text("contract"), policy.strings("allowed_lineages"), policy.strings("accelerators"), policy.optional("backend"))
            val active = case["active"]?.takeUnless { it.isJsonNull }?.asJsonObject
            val result = resolveCatalog(catalog, case.text("capability"), profile, request, active)
            val expected = case.child("expected")
            for (release in catalog.releases) assertEquals(case.text("name"), expected.child("release_digests").text(release.id), release.digest)
            assertEquals(case.text("name"), expected.text("device_fingerprint"), qualificationFingerprint(profile))
            assertEquals(case.text("name"), expected.text("status"), result.text("status"))
            assertEquals(case.text("name"), expected.optional("variant"), result.child("selection").optional("variant"))
            assertEquals(case.text("name"), expected["app_update_available"].asBoolean, result["app_update_available"].asBoolean)
        }
    }

    @Test fun exactChipAndVersionBoundaries() {
        assertEquals(canonicalSoc("MediaTek MT6991"), canonicalSoc("MT6991"))
        assertNotEquals(canonicalSoc("MT6991"), canonicalSoc("MT69910"))
        assertEquals(canonicalSoc("Tensor G6"), canonicalSoc("Google Tensor G6"))
        assertTrue(versionMatches("2.3.0", ">=2.3,<3"))
        assertFalse(versionMatches("3.0.0", ">=2.3,<3"))
        assertFalse(versionMatches(null, ">=2.3,<3"))
    }

    @Test fun executableArtifactsAndUnsignedCatalogsAreRejected() {
        val raw = fixtures.array("cases")[0].asJsonObject.child("catalog").deepCopy()
        try {
            decodeCatalog(json.toJson(raw).toByteArray(), CatalogTrust(emptyMap()), false)
            fail("Unsigned model catalogs must be explicitly enabled")
        } catch (fault: RuntimeFault) { assertEquals("untrusted_catalog", fault.code) }
        raw.array("releases")[0].asJsonObject.child("manifest").child("artifacts").child("vision.executorch").addProperty("path", "downloaded-code.so")
        try {
            decodeCatalog(json.toJson(raw).toByteArray(), CatalogTrust(emptyMap()), true)
            fail("Model-data delivery must not become native-code delivery")
        } catch (fault: RuntimeFault) { assertEquals("invalid_catalog", fault.code) }
    }

    @Test fun duplicateKeysAndNonFiniteNumbersCannotChangeContractMeaning() {
        for (value in listOf("{\"a\":1,\"a\":2}", "{\"a\":NaN}", "{\"a\":1e999}", "{\"a\":1}{}")) {
            try { parseObject(value.toByteArray()); fail("Invalid JSON accepted: $value") }
            catch (_: Exception) { /* Expected; Python rejects the same encodings. */ }
        }
    }

    @Test fun referenceBytesMustMatchTheDeclaredTensor() {
        val raw = fixtures.array("cases")[0].asJsonObject.child("catalog").deepCopy()
        raw.array("releases")[0].asJsonObject.child("manifest").child("artifacts").child("qualification.output").addProperty("size", 4004)
        try { decodeCatalog(json.toJson(raw).toByteArray(), CatalogTrust(emptyMap()), true); fail("Invalid tensor reference accepted") }
        catch (fault: RuntimeFault) { assertEquals("invalid_catalog", fault.code) }
    }

    @Test fun numericSpellingDoesNotBreakInstalledReleaseIdentity() {
        val raw = parseObject("{\"metric\":3.3855438232421875e-05,\"tiny\":1e-7}".toByteArray())
        assertEquals("{\"metric\":3.3855438232421875e-05,\"tiny\":1e-7}", canonical(raw))
        val releaseJson = fixtures.array("cases")[0].asJsonObject.child("catalog").array("releases")[0].asJsonObject.deepCopy()
        releaseJson.child("manifest").add("metadata", raw)
        val release = ModelRelease(releaseJson)
        val legacy = sha256(canonical(releaseJson, legacyNumbers = true).toByteArray())
        assertEquals(legacy, release.atInstalledDigest(legacy, "release_conflict").digest)
        assertEquals(canonical(releaseJson, legacyNumbers = true), canonical(parseObject(canonical(releaseJson, legacyNumbers = true).toByteArray()), legacyNumbers = true))
        try { release.atInstalledDigest("0".repeat(64), "release_conflict"); fail("Unrelated identity must remain rejected") }
        catch (fault: RuntimeFault) { assertEquals("release_conflict", fault.code) }
    }
}
