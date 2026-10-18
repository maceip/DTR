package ai.phaseone.demo

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.ScrollView
import android.widget.TextView
import ai.phaseone.runtime.CatalogTrust
import ai.phaseone.runtime.PhaseoneRuntime
import ai.phaseone.runtime.ReleasePolicy
import com.google.gson.JsonObject
import java.util.concurrent.Executors

/** One application flow; all device/backend decisions belong to the runtime. */
class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var runtime: PhaseoneRuntime
    private lateinit var status: TextView
    private lateinit var result: TextView
    private lateinit var detail: TextView
    private lateinit var run: Button
    private lateinit var menu: Button
    private val policy = ReleasePolicy("imagenet1k.rgb224.v1", listOf("torchvision.mobilenet-v3-small.imagenet1k-v1"))
    private val catalog = "http://127.0.0.1:8930/catalog.json"
    private val ink by lazy { getColor(R.color.phaseone_ink) }
    private val paper by lazy { getColor(R.color.phaseone_paper) }
    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val key = assets.open("catalog-publisher.pub").bufferedReader().use { it.readText().trim() }.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
        runtime = PhaseoneRuntime(this, CatalogTrust.key(key), localDevelopment = true)
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setPadding(dp(24), dp(36), dp(24), dp(24)); setBackgroundColor(paper)
        }
        val scroll = ScrollView(this).apply { setBackgroundColor(paper); addView(layout) }
        setContentView(scroll)
        // Insets remain visible on both gesture-navigation phones.
        scroll.setOnApplyWindowInsetsListener { view, insets ->
            val bars = insets.getInsets(android.view.WindowInsets.Type.systemBars())
            view.setPadding(0, bars.top, 0, bars.bottom); insets
        }
        fun text(value: String, size: Float, bold: Boolean = false): TextView = TextView(this).apply {
            text = value; textSize = size; setTextColor(ink)
            if (bold) typeface = Typeface.create("sans-serif", Typeface.BOLD)
        }
        layout.addView(text("PHASEONE  [tpu]", 17f, true))
        layout.addView(text("RUNTIME LAB", 11f).apply { letterSpacing = .18f; setPadding(0, dp(8), 0, dp(28)) })
        layout.addView(text("One app.\nNative everywhere.", 39f, true).apply { setLineSpacing(0f, .96f) })
        layout.addView(text("A local model, selected for this device.", 16f).apply { setPadding(0, dp(16), 0, dp(24)) })
        val device = runtime.device
        val deviceCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setPadding(dp(18), dp(16), dp(18), dp(16))
            background = GradientDrawable().apply { setColor(getColor(R.color.phaseone_card)); cornerRadius = dp(16).toFloat() }
        }
        deviceCard.addView(text(device.model, 18f, true))
        deviceCard.addView(text("${device.soc} · Android ${android.os.Build.VERSION.RELEASE}", 13f).apply { setPadding(0, dp(4), 0, 0) })
        layout.addView(deviceCard)
        status = text("Ready for local inference", 13f).apply { setPadding(0, dp(22), 0, dp(12)) }
        layout.addView(status)
        val preview = ImageView(this).apply {
            setImageDrawable(android.graphics.drawable.Drawable.createFromStream(assets.open("sample-dog.jpg"), null))
            scaleType = ImageView.ScaleType.CENTER_CROP
            background = GradientDrawable().apply { cornerRadius = dp(16).toFloat() }
            clipToOutline = true
        }
        layout.addView(preview, LinearLayout.LayoutParams(-1, dp(190)))
        result = text("Run the same image model", 24f, true).apply { setPadding(0, dp(20), 0, dp(8)) }
        detail = text("This sample checks model delivery and execution. Inference runs inside this app.", 14f).apply { setLineSpacing(dp(3).toFloat(), 1f) }
        layout.addView(result); layout.addView(detail)
        run = Button(this).apply {
            text = "Run locally"; isAllCaps = false; textSize = 16f; setTextColor(getColor(R.color.phaseone_on_accent))
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.rgb(193, 246, 92))
            setOnClickListener { perform { runLocal() } }
        }
        layout.addView(run, LinearLayout.LayoutParams(-1, dp(58)).apply { topMargin = dp(24) })
        menu = Button(this).apply {
            text = "Model releases  ···"; isAllCaps = false; setTextColor(ink)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.TRANSPARENT)
            setOnClickListener { anchor -> showReleaseMenu(anchor) }
        }
        layout.addView(menu, LinearLayout.LayoutParams(-1, dp(48)))
        layout.addView(text("ExecuTorch + LiteRT\nModel downloads are data. Native engines ship with the app.", 12f).apply {
            setPadding(0, dp(18), 0, 0); setTextColor(getColor(R.color.phaseone_muted)); gravity = Gravity.CENTER
        })
    }

    private fun showReleaseMenu(anchor: View) {
        PopupMenu(this, anchor).apply {
            menu.add("Check and install updates")
            menu.add("Roll back model release")
            setOnMenuItemClickListener { item ->
                perform {
                    val response = if (item.title == "Roll back model release") runtime.releases.rollback("vision")
                    else runtime.releases.update(catalog, "vision", policy) { update -> runOnUiThread { status.text = update } }
                    runOnUiThread { status.text = response["status"].asString.replace('_', ' ') }
                    runLocal()
                }; true
            }; show()
        }
    }

    private fun runLocal() {
        val state = runtime.releases.state("vision")
        if (state.get("active") == null || state["active"].isJsonNull) {
            runtime.releases.update(catalog, "vision", policy) { update -> runOnUiThread { status.text = update } }
        }
        val image = assets.open("sample-dog.jpg").use { it.readBytes() }
        val classified = runtime.open("vision").use { it.classify("vision", image) }
        val selected = runtime.releases.state("vision")["active"].asJsonObject
        val execution = classified.execution
        runOnUiThread {
            result.text = classified.predictions.first().label
            status.text = "Executed locally · ${"%.1f".format(execution.inference_ms)} ms"
            detail.text = "${execution.backend} · ${execution.accelerator_observed ?: "acceleration unverified"}\n${selected["release"].asString}\nTask checks passed on this device."
        }
    }

    private fun perform(action: () -> Unit) {
        run.isEnabled = false; menu.isEnabled = false; status.text = "Working on this device…"
        worker.execute {
            try { action() }
            catch (failure: Throwable) { runOnUiThread { status.text = "Action needs attention"; detail.text = failure.message ?: failure.javaClass.simpleName } }
            finally { runOnUiThread { run.isEnabled = true; menu.isEnabled = true } }
        }
    }

    override fun onDestroy() { worker.shutdown(); super.onDestroy() }
}
