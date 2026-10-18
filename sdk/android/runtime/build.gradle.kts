import java.security.MessageDigest

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("maven-publish")
}
val runtimeSources = fileTree("src/main") { include("**/*.kt", "**/*.xml") }.files.sortedBy { it.relativeTo(projectDir).path }
val runtimeHasher = MessageDigest.getInstance("SHA-256")
for (source in runtimeSources + file("build.gradle.kts")) {
    runtimeHasher.update(source.relativeTo(projectDir).path.toByteArray())
    runtimeHasher.update(source.readBytes())
}
val runtimeBuildId = runtimeHasher.digest().joinToString("") { "%02x".format(it) }
android {
    namespace = "ai.phaseone.runtime"
    compileSdk = 37
    buildFeatures { buildConfig = true }
    defaultConfig {
        minSdk = 31
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "RUNTIME_BUILD_ID", "\"$runtimeBuildId\"")
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    packaging { jniLibs { pickFirsts += "**/libc++_shared.so" } }
    publishing { singleVariant("release") { withSourcesJar() } }
}
kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }
dependencies {
    // Device metadata and qualified artifacts refer to this exact native ABI set.
    // A consumer's transitive dependency must not silently replace an engine.
    implementation("org.pytorch:executorch-android:1.5.1") { version { strictly("1.5.1") } }
    implementation("com.google.ai.edge.litert:litert:2.3.0") { version { strictly("2.3.0") } }
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.18.0") { version { strictly("0.18.0") } }
    implementation("com.google.ai.edge.litert:litert-npu-runtime-google-tensor:2.3.0") { version { strictly("2.3.0") } }
    implementation("com.google.ai.edge.litert:litert-npu-runtime-mediatek:2.3.0") { version { strictly("2.3.0") } }
    api("com.google.code.gson:gson:2.14.0")
    testImplementation("junit:junit:4.13.2")
}
afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])
                groupId = "ai.phaseone"
                artifactId = "runtime-android"
                version = "0.3.0"
                pom { name.set("PHASEONE Runtime"); description.set("Local AI applications across native ExecuTorch and LiteRT engines") }
            }
        }
        repositories { maven { name = "distribution"; url = uri("../../../dist/android-maven") } }
    }
}
