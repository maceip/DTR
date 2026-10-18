plugins { id("com.android.library") }
android {
    namespace = "ai.phaseone.runtime.portable"
    compileSdk = 37
    defaultConfig { minSdk = 33 }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
dependencies {
    implementation("com.dylibso.chicory:runtime:1.7.5") { version { strictly("1.7.5") } }
    testImplementation("junit:junit:4.13.2")
    testImplementation("com.dylibso.chicory:wabt:1.7.5")
}
