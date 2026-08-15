// Motion Vectors - the launcher and configuration app.
//
// Two jobs. It sets the two Vulkan loader variables and starts X-Plane, and it
// is where every setting lives.
//
// The loader variables are what make the layer EXPLICIT. Registering it through
// the loader's ImplicitLayers key would load the DLL into every Vulkan
// application on the machine; setting them here scopes it to the one process
// this launcher starts, and nothing else on the system is touched.
//
// ---- WHY UNFINISHED BACKENDS ARE SHOWN RATHER THAN HIDDEN.
//
// Every backend this project intends to support appears in the list, and the
// ones that do not work yet say so on their own row. Hiding them would make the
// app look finished and leave a user wondering whether DLSS is missing because
// their card lacks it, because the install is broken, or because it was never
// written - three very different problems that look identical when the option
// simply is not there. A row that says "in progress" answers that in one glance.
//
// On a release build such a row is disabled, so it cannot be chosen by accident.
// On a development build it stays selectable, because a backend nobody can
// select is a backend nobody can test while it is being written. See devBuild().

#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>
#include <QSlider>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QPlainTextEdit>
#include <QTabWidget>
#include <QProcess>
#include <QDir>
#include <QFileInfo>
#include <QFileDialog>
#include <QSettings>
#include <QStandardItemModel>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDesktopServices>
#include <QUrl>

#include "../mv_icon.h"

// ---- THE VERSION COMES FROM THE BUILD, NOT FROM A LITERAL HERE.
//
// This was `static const char *kVersion = "0.0.08"` while the build was 0.0.16,
// so the window title, the update check and any bug report made from it all
// named a release eight versions old. build.ps1 has been passing -DMV_VERSION
// from the VERSION file the whole time; the launcher simply never read it. That
// is exactly the disagreement the VERSION file was introduced to end, and it
// survived here because until now the app crashed before anyone could read the
// title.
#ifndef MV_VERSION
#define MV_VERSION "dev"
#endif
static const char *kVersion  = MV_VERSION;

// ---- ONE BINARY, TWO FACES. THE VERSION STRING DECIDES WHICH.
//
// A build whose version is "dev" opens every developer surface: the Advanced
// tab, the debug console, the layer log, the saved-situation shortcut, and the
// backends that are not finished - selectable, so they can be tested as they are
// written. Any other version is a RELEASE and shows none of it.
//
// The switch is the version rather than a separate define on purpose. A build
// that says 0.0.16 and behaves like a development build is the same class of
// problem as a launcher that said 0.0.08 while the build was 0.0.16 - two facts
// about one binary that disagree. Tying them to a single string makes that
// impossible: if it claims to be a release, it IS one.
//
// build.ps1 -Dev sets it. VERSION supplies it otherwise.
static bool devBuild()
{
    static const bool on = QString(kVersion).contains("dev", Qt::CaseInsensitive);
    return on;
}
static const char *kLayer    = "VK_LAYER_mv";
static const char *kReleases =
    "https://api.github.com/repos/Vihaan2012-cmyk/Motion-Vectors-for-Xplane-12-V2/releases/latest";

// Every backend, with what it actually is today.
//
// `env` is the value passed as TAA_BACKEND / TAA_FRAMEGEN. `ready` is whether
// selecting it does anything. Keeping the two in one table means the UI cannot
// disagree with what the layer will accept.
struct Backend {
    const char *label;
    const char *env;
    bool        ready;
    const char *note;
};

static const Backend kReconstruction[] = {
    { "Off - no temporal pass",              "off",  true,
      "X-Plane renders and presents exactly as it would without this installed." },
    { "TAA - temporal antialiasing (ours)",  "taa",  true,
      "Native resolution. The only backend that needs no vendor SDK, so it is "
      "the fallback that always exists." },
    { "TAAU - temporal upscaling (ours)",    "taau", false,
      "Renders below display resolution and reconstructs. Next after TAA." },
    { "DLAA - NVIDIA, native resolution",    "dlaa", false,
      "The DLSS network at render = display. Wired first of the vendor "
      "backends because it has no resolution plumbing to get wrong." },
    { "DLSS Super Resolution - NVIDIA",      "dlss", false, "Needs the DLSS SDK." },
    { "FSR - AMD FidelityFX 2/3",            "fsr",  false,
      "Runs on any vendor, not only AMD." },
    { "FSR 4 - AMD, machine learning",       "fsr4", false,
      "Needs RDNA4-class matrix hardware; the SDK reports whether it is present." },
    { "XeSS - Intel",                        "xess", false,
      "Runs on any vendor. Expects a different motion-vector convention to "
      "ours, which the backend converts." },
    { "NIS - NVIDIA Image Scaling (spatial)", "nis", false,
      "Spatial only: no motion vectors, no history. Sits below the temporal "
      "path entirely." },
};

static const Backend kFrameGen[] = {
    { "Off",                                  "off",  true,
      "Recommended for now. Frame generation interpolates, so anything that "
      "does not move with the scene - the panel, the instruments - needs "
      "masking out first." },
    { "FSR Frame Generation - AMD",           "fsr",  false, "Needs the FSR 3 SDK." },
    { "DLSS Frame Generation - NVIDIA",       "dlss", false, "Ada-class or newer." },
    { "DLSS Multi Frame Generation - NVIDIA", "mfg",  false, "Blackwell-class." },
    { "XeSS Frame Generation - Intel",        "xess", false, "Needs the XeSS SDK." },
};

// Render scale presets, named the way every other title names them so the
// numbers are not a surprise.
struct QualityPreset { const char *label; float scale; };
static const QualityPreset kQuality[] = {
    { "Native - 100%",            1.00f },
    { "Quality - 67%",            0.667f },
    { "Balanced - 58%",           0.58f },
    { "Performance - 50%",        0.50f },
    { "Ultra performance - 33%",  0.333f },
};

class Launcher : public QWidget {
public:
    Launcher()
    {
        setWindowTitle(devBuild()
            ? QString("Motion Vectors  %1   [DEVELOPER BUILD]").arg(kVersion)
            : QString("Motion Vectors  %1").arg(kVersion));
        setWindowIcon(mvIcon());
        setMinimumWidth(680);

        auto *root = new QVBoxLayout(this);

        m_status = new QLabel;
        m_status->setWordWrap(true);
        root->addWidget(m_status);

        m_path = new QLabel;
        m_path->setWordWrap(true);
        m_path->setStyleSheet("color:#888");
        root->addWidget(m_path);

        auto *row = new QHBoxLayout;
        auto *browse = new QPushButton("Choose X-Plane folder...");
        connect(browse, &QPushButton::clicked, this, &Launcher::browse);
        row->addWidget(browse);
        row->addStretch();
        root->addLayout(row);

        auto *tabs = new QTabWidget;
        tabs->addTab(buildRendering(), "Rendering");
        tabs->addTab(buildQuality(),   "Quality");
        // Developer surfaces only exist on a dev build. Hidden rather than
        // disabled here, because unlike the unfinished backends there is nothing
        // for a user to understand from their absence - they are not features
        // that are coming, they are instrumentation.
        if (devBuild()) tabs->addTab(buildAdvanced(), "Advanced");
        root->addWidget(tabs);

        m_launch = new QPushButton("Launch X-Plane 12");
        m_launch->setMinimumHeight(40);
        connect(m_launch, &QPushButton::clicked, this, &Launcher::launch);
        root->addWidget(m_launch);

        m_update = new QLabel;
        m_update->setOpenExternalLinks(true);
        m_update->setWordWrap(true);
        root->addWidget(m_update);

        m_log = new QPlainTextEdit;
        m_log->setReadOnly(true);
        m_log->setMaximumHeight(80);
        root->addWidget(m_log);

        QSettings s("MotionVectors", "Launcher");
        m_root = s.value("xplane").toString();
        if (m_root.isEmpty() || !hasXPlane(m_root)) m_root = guessRoot();
        loadSettings();

        refresh();
        checkForUpdate();
    }

private:
    // ---------------------------------------------------------- rendering tab
    QWidget *buildRendering()
    {
        auto *w = new QWidget;
        auto *v = new QVBoxLayout(w);

        auto *g1 = new QGroupBox("Antialiasing and upscaling");
        auto *f1 = new QFormLayout(g1);
        m_recon = new QComboBox;
        fill(m_recon, kReconstruction,
             sizeof(kReconstruction) / sizeof(kReconstruction[0]));
        f1->addRow("Backend", m_recon);
        m_reconNote = new QLabel;
        m_reconNote->setWordWrap(true);
        m_reconNote->setStyleSheet("color:#888");
        f1->addRow(m_reconNote);
        connect(m_recon, QOverload<int>::of(&QComboBox::currentIndexChanged),
                [this](int i){
                    if (i >= 0) m_reconNote->setText(kReconstruction[i].note);
                });
        v->addWidget(g1);

        auto *g2 = new QGroupBox("Frame generation");
        auto *f2 = new QFormLayout(g2);
        m_fg = new QComboBox;
        fill(m_fg, kFrameGen, sizeof(kFrameGen) / sizeof(kFrameGen[0]));
        f2->addRow("Backend", m_fg);
        m_fgNote = new QLabel;
        m_fgNote->setWordWrap(true);
        m_fgNote->setStyleSheet("color:#888");
        f2->addRow(m_fgNote);
        connect(m_fg, QOverload<int>::of(&QComboBox::currentIndexChanged),
                [this](int i){ if (i >= 0) m_fgNote->setText(kFrameGen[i].note); });
        v->addWidget(g2);

        // Frame generation is a SEPARATE axis from reconstruction, not a
        // flavour of it, so any supported pairing composes - DLSS upscaling with
        // FSR frame generation, for instance. Saying so here stops the two
        // dropdowns reading as one choice split across two rows.
        auto *axis = new QLabel(
            "<i>These are independent. Any supported pairing works - an upscaler "
            "from one vendor with frame generation from another is a normal "
            "configuration, not a workaround.</i>");
        axis->setWordWrap(true);
        v->addWidget(axis);

        auto *g3 = new QGroupBox("X-Plane's own antialiasing");
        auto *f3 = new QVBoxLayout(g3);
        m_killFxaa = new QCheckBox("Turn off X-Plane's FXAA while a temporal backend runs");
        m_killMsaa = new QCheckBox("Turn off X-Plane's MSAA while a temporal backend runs");
        f3->addWidget(m_killFxaa);
        f3->addWidget(m_killMsaa);
        auto *why = new QLabel(
            "<i>X-Plane composites tonemap, bloom and FXAA in one pass that runs "
            "<b>after</b> our resolve, so its FXAA blurs a frame that is already "
            "antialiased. MSAA is worse than redundant: it produces a "
            "multisampled target the resolve cannot use, so leaving it on does "
            "not degrade the result, it disables it. Both are restored when the "
            "plugin unloads.</i>");
        why->setWordWrap(true);
        f3->addWidget(why);
        v->addWidget(g3);

        v->addStretch();
        return w;
    }

    // ------------------------------------------------------------ quality tab
    QWidget *buildQuality()
    {
        auto *w = new QWidget;
        auto *v = new QVBoxLayout(w);

        auto *g1 = new QGroupBox("Render resolution");
        auto *f1 = new QFormLayout(g1);
        m_quality = new QComboBox;
        for (const QualityPreset &q : kQuality) m_quality->addItem(q.label);
        f1->addRow("Preset", m_quality);
        m_scale = new QDoubleSpinBox;
        m_scale->setRange(0.25, 1.0);
        m_scale->setSingleStep(0.01);
        m_scale->setDecimals(3);
        f1->addRow("Render scale", m_scale);
        connect(m_quality, QOverload<int>::of(&QComboBox::currentIndexChanged),
                [this](int i){ if (i >= 0) m_scale->setValue(kQuality[i].scale); });
        auto *n1 = new QLabel(
            "<i>Below 100% only has an effect once an upscaling backend is "
            "running. With TAA, which is native-resolution, the sim simply "
            "renders smaller and the result is softer - so this stays at Native "
            "until TAAU or a vendor upscaler is selected.</i>");
        n1->setWordWrap(true);
        f1->addRow(n1);
        v->addWidget(g1);

        auto *g2 = new QGroupBox("Sharpening");
        auto *f2 = new QFormLayout(g2);
        m_sharp = new QDoubleSpinBox;
        m_sharp->setRange(0.0, 1.0);
        m_sharp->setSingleStep(0.05);
        m_sharp->setDecimals(2);
        f2->addRow("Amount", m_sharp);
        f2->addRow(inProgress("Applied by the backend's own sharpening pass, so "
                              "it takes effect when a vendor backend is wired."));
        v->addWidget(g2);

        auto *g3 = new QGroupBox("Textures");
        auto *f3 = new QFormLayout(g3);
        m_texRes = new QComboBox;
        // Straight from the binary: reno/tex_res maps to a linear scale on
        // texture dimensions, and these are the exact factors it uses.
        m_texRes->addItem("Maximum - full size (5)");
        m_texRes->addItem("Very high - 1/2 (4)");
        m_texRes->addItem("High - 1/4 (3)");
        m_texRes->addItem("Medium - 1/8 (2)");
        m_texRes->addItem("Low - 1/16 (1)");
        m_texRes->addItem("Minimum - 1/32 (0)");
        f3->addRow("Quality", m_texRes);
        m_pager = new QCheckBox("Use the texture pager (keeps VRAM under a headroom target)");
        f3->addRow(m_pager);
        m_headroom = new QSpinBox;
        m_headroom->setRange(0, 8192);
        m_headroom->setSuffix(" MB");
        f3->addRow("VRAM headroom", m_headroom);
        v->addWidget(g3);

        v->addStretch();
        return w;
    }

    // ----------------------------------------------------------- advanced tab
    QWidget *buildAdvanced()
    {
        auto *w = new QWidget;
        auto *v = new QVBoxLayout(w);

        auto *g1 = new QGroupBox("Diagnostics");
        auto *f1 = new QVBoxLayout(g1);
        m_trace = new QCheckBox("Write the layer log (slower - it logs from the render path)");
        f1->addWidget(m_trace);
        m_situation = new QCheckBox("Start straight into the saved situation (debugging)");
        m_situation->setToolTip(
            "Uses --load_smo. It is the only option that bypasses the flight "
            "configuration screen: --load_acf, --load_apt, --go_to_apt and "
            "--new_flight_json are all parsed but only pre-fill it.");
        f1->addWidget(m_situation);

        auto *dbgRow = new QHBoxLayout;
        auto *dbg = new QPushButton("Open the Debug Console");
        connect(dbg, &QPushButton::clicked, this, &Launcher::openDebug);
        dbgRow->addWidget(dbg);
        dbgRow->addStretch();
        f1->addLayout(dbgRow);
        auto *dbgNote = new QLabel(
            "<i>The console runs beside the sim. Everything it changes takes "
            "effect on the next frame, so tuning and diagnosis never need a "
            "restart.</i>");
        dbgNote->setWordWrap(true);
        f1->addWidget(dbgNote);
        v->addWidget(g1);

        auto *g2 = new QGroupBox("Where things are");
        auto *f2 = new QFormLayout(g2);
        m_layerPath = new QLabel;
        m_layerPath->setTextInteractionFlags(Qt::TextSelectableByMouse);
        m_layerPath->setWordWrap(true);
        f2->addRow("Layer", m_layerPath);
        auto *live = new QLabel(liveFilePath());
        live->setTextInteractionFlags(Qt::TextSelectableByMouse);
        live->setWordWrap(true);
        f2->addRow("Live controls", live);
        v->addWidget(g2);

        v->addStretch();
        return w;
    }

    QLabel *inProgress(const QString &s)
    {
        auto *l = new QLabel("<b style='color:#b26a00'>In progress.</b> <i>" + s + "</i>");
        l->setWordWrap(true);
        return l;
    }

    // Populate a combo from a backend table, disabling the rows that are not
    // wired yet while leaving them visible and readable.
    void fill(QComboBox *c, const Backend *b, int n)
    {
        for (int i = 0; i < n; ++i) {
            c->addItem(b[i].ready ? QString(b[i].label)
                                  : QString("%1   -   in progress").arg(b[i].label));
            if (b[i].ready) continue;
            // On a RELEASE build the row is disabled: visible, legible, and
            // impossible to choose. Showing it matters - an option that is
            // simply absent leaves a user unable to tell "my card lacks it" from
            // "the install is broken" from "it was never written", and those are
            // three different problems.
            //
            // On a DEV build it stays selectable, because a backend that cannot
            // be chosen cannot be tested while it is being written.
            if (devBuild()) continue;
            if (auto *m = qobject_cast<QStandardItemModel*>(c->model()))
                if (QStandardItem *it = m->item(i))
                    it->setFlags(it->flags() & ~Qt::ItemIsEnabled);
        }
    }

    static bool hasXPlane(const QString &dir)
    {
        return QFileInfo::exists(QDir(dir).filePath("X-Plane.exe"));
    }

    QString guessRoot() const
    {
        QDir here(QCoreApplication::applicationDirPath());
        if (hasXPlane(here.absolutePath())) return here.absolutePath();
        // Walk up: installed we sit two deep (MotionVectors\launcher), and in
        // the source tree three (MotionVectors\build\qtlauncher). Climbing until
        // X-Plane.exe appears covers both without hard-coding either.
        for (int i = 0; i < 4; ++i) {
            if (!here.cdUp()) break;
            if (hasXPlane(here.absolutePath())) return here.absolutePath();
        }
        const QStringList guesses = {
            "D:/Steam Games/steamapps/common/X-Plane 12",
            "C:/Program Files/X-Plane 12",
            "C:/X-Plane 12",
        };
        for (const QString &g : guesses)
            if (hasXPlane(g)) return g;
        return QString();
    }

    // ---- FIND THE MANIFEST WHEREVER WE ARE RUN FROM.
    //
    // This looked beside the executable and then in "<us>/build/vklayer", which
    // is right when installed and wrong in the source tree - there the launcher
    // is ALREADY inside build/, so the layer is a sibling at ../vklayer and the
    // app reported "Layer not found - reinstall" on a perfectly good build.
    QString layerDir() const
    {
        QDir d(QCoreApplication::applicationDirPath());
        const QStringList tries = {
            ".",                 // installed: beside us
            "../vklayer",        // source tree: build/qtlauncher -> build/vklayer
            "build/vklayer",     // run from the project root
            "../../build/vklayer",
        };
        for (const QString &t : tries) {
            const QString p = QDir::cleanPath(d.filePath(t));
            if (QFileInfo::exists(QDir(p).filePath("VkLayer_mv.json")))
                return p;
        }
        return QDir::cleanPath(d.filePath("../vklayer"));
    }

    QString liveFilePath() const
    {
        const QByteArray t = qgetenv("TEMP");
        return QString::fromLocal8Bit(t.isEmpty() ? "." : t) + "\\taa_live.ini";
    }

    void openDebug()
    {
        const QString exe = QDir(QCoreApplication::applicationDirPath())
                                .filePath("MotionVectorsDebug.exe");
        if (!QFileInfo::exists(exe)) {
            m_log->appendPlainText("Debug console not found beside this app: " + exe);
            return;
        }
        QProcess::startDetached(exe, {}, QCoreApplication::applicationDirPath());
    }

    void refresh()
    {
        const bool okXP = !m_root.isEmpty() && hasXPlane(m_root);
        const QString manifest = QDir(layerDir()).filePath("VkLayer_mv.json");
        const bool okLayer = QFileInfo::exists(manifest);

        QString s;
        if (!okXP)
            s = "<b style='color:#c0392b'>X-Plane 12 not found.</b> Choose the folder that contains X-Plane.exe.";
        else if (!okLayer)
            s = "<b style='color:#c0392b'>Layer not found.</b> VkLayer_mv.json is missing from " +
                layerDir() + " - reinstall, or run build.ps1 if this is the source tree.";
        else
            s = "<b style='color:#1e8449'>Ready.</b> The layer is enabled for X-Plane only, "
                "so no other Vulkan application loads it.";
        m_status->setText(s);
        m_path->setText(okXP ? QString("X-Plane: %1<br>Layer: %2").arg(m_root, layerDir())
                             : QString("Layer: %1").arg(layerDir()));
        if (m_layerPath) m_layerPath->setText(layerDir());
        m_launch->setEnabled(okXP && okLayer);
    }

    void browse()
    {
        const QString d = QFileDialog::getExistingDirectory(this, "Select your X-Plane 12 folder", m_root);
        if (d.isEmpty()) return;
        if (!hasXPlane(d)) {
            m_log->appendPlainText("That folder has no X-Plane.exe: " + d);
            return;
        }
        m_root = d;
        QSettings("MotionVectors", "Launcher").setValue("xplane", m_root);
        refresh();
    }

    void loadSettings()
    {
        QSettings s("MotionVectors", "Launcher");
        m_recon->setCurrentIndex(s.value("recon", 1).toInt());       // TAA
        m_fg->setCurrentIndex(s.value("framegen", 0).toInt());
        m_quality->setCurrentIndex(s.value("quality", 0).toInt());
        m_scale->setValue(s.value("scale", 1.0).toDouble());
        m_sharp->setValue(s.value("sharpness", 0.5).toDouble());
        m_texRes->setCurrentIndex(s.value("texres", 0).toInt());
        m_pager->setChecked(s.value("pager", true).toBool());
        m_headroom->setValue(s.value("headroom", 200).toInt());
        m_killFxaa->setChecked(s.value("killfxaa", true).toBool());
        m_killMsaa->setChecked(s.value("killmsaa", true).toBool());
        if (m_trace)     m_trace->setChecked(s.value("trace", false).toBool());
        if (m_situation) m_situation->setChecked(s.value("situation", false).toBool());
    }

    void saveSettings()
    {
        QSettings s("MotionVectors", "Launcher");
        s.setValue("recon",     m_recon->currentIndex());
        s.setValue("framegen",  m_fg->currentIndex());
        s.setValue("quality",   m_quality->currentIndex());
        s.setValue("scale",     m_scale->value());
        s.setValue("sharpness", m_sharp->value());
        s.setValue("texres",    m_texRes->currentIndex());
        s.setValue("pager",     m_pager->isChecked());
        s.setValue("headroom",  m_headroom->value());
        s.setValue("killfxaa",  m_killFxaa->isChecked());
        s.setValue("killmsaa",  m_killMsaa->isChecked());
        if (m_trace)     s.setValue("trace",     m_trace->isChecked());
        if (m_situation) s.setValue("situation", m_situation->isChecked());
    }

    void launch()
    {
        saveSettings();

        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        env.insert("VK_LAYER_PATH", QDir::toNativeSeparators(layerDir()));
        env.insert("VK_LOADER_LAYERS_ENABLE", kLayer);

        const Backend &rb = kReconstruction[m_recon->currentIndex()];
        const Backend &fb = kFrameGen[m_fg->currentIndex()];
        env.insert("TAA_BACKEND",  rb.env);
        env.insert("TAA_FRAMEGEN", fb.env);
        // TAA_RESOLVE is what the layer's own gate reads; the backend name is
        // carried alongside it so the two cannot drift as more are added.
        if (QString(rb.env) != "off") env.insert("TAA_RESOLVE", "1");

        env.insert("TAA_RENDER_SCALE", QString::number(m_scale->value(), 'f', 4));
        env.insert("TAA_SHARPNESS",    QString::number(m_sharp->value(), 'f', 3));
        // The combo runs Maximum..Minimum, and reno/tex_res runs 5..0, so the
        // index is inverted rather than mapped through a table.
        env.insert("TAA_TEX_RES", QString::number(5 - m_texRes->currentIndex()));
        if (m_pager->isChecked()) {
            env.insert("TAA_PAGER_DROP_ABOVE", "2048");
            env.insert("TAA_PAGER_MAX_DROP",   "1");
            env.insert("TAA_PAGER_HEADROOM_MB", QString::number(m_headroom->value()));
        }
        // These two are opt-OUT in the plugin, so the launcher sets the keep
        // flags rather than the kill flags. Same meaning, opposite polarity, and
        // worth stating because the checkbox reads the other way round.
        if (!m_killFxaa->isChecked()) env.insert("TAA_KEEP_FXAA", "1");
        if (!m_killMsaa->isChecked()) env.insert("TAA_KEEP_MSAA", "1");
        if (m_trace && m_trace->isChecked()) env.insert("TAA_LAYER_TRACE", "1");

        QStringList args;
        if (m_situation && m_situation->isChecked())
            args << "--load_smo=Output/situations/Cirrus SR-22 SituationV2.sit";

        auto *p = new QProcess(this);
        p->setProcessEnvironment(env);
        p->setWorkingDirectory(m_root);
        p->setProgram(QDir(m_root).filePath("X-Plane.exe"));
        p->setArguments(args);

        if (!p->startDetached()) {
            m_log->appendPlainText("Could not start X-Plane.");
            return;
        }
        m_log->appendPlainText(QString("Started  backend=%1  framegen=%2  scale=%3%4")
                                   .arg(rb.env, fb.env)
                                   .arg(m_scale->value(), 0, 'f', 3)
                                   .arg(args.isEmpty() ? "" : "  " + args.join(' ')));
    }

    // Reports a newer release; it does not download or replace anything. An
    // updater that rewrites files while the sim may be running is a bigger
    // promise than this needs to make.
    void checkForUpdate()
    {
        auto *nam = new QNetworkAccessManager(this);
        QNetworkRequest req{QUrl(kReleases)};
        req.setRawHeader("Accept", "application/vnd.github+json");
        req.setRawHeader("User-Agent", "MotionVectorsLauncher");
        connect(nam, &QNetworkAccessManager::finished, this, [this](QNetworkReply *r) {
            r->deleteLater();
            if (r->error() != QNetworkReply::NoError) {
                m_update->setText("<span style='color:#888'>Update check failed.</span>");
                return;
            }
            const QJsonObject o = QJsonDocument::fromJson(r->readAll()).object();
            const QString tag = o.value("tag_name").toString();
            if (tag.isEmpty()) return;
            if (tag == QString(kVersion) || tag == QString("v") + kVersion) {
                m_update->setText("<span style='color:#888'>Up to date (" + tag + ").</span>");
            } else {
                m_update->setText(
                    "A newer build is available: <b>" + tag + "</b> &mdash; "
                    "<a href='" + o.value("html_url").toString() + "'>open the release page</a>");
            }
        });
        nam->get(req);
    }

    QString m_root;
    QLabel *m_status = nullptr, *m_path = nullptr, *m_update = nullptr;
    QLabel *m_reconNote = nullptr, *m_fgNote = nullptr, *m_layerPath = nullptr;
    QComboBox *m_recon = nullptr, *m_fg = nullptr,
              *m_quality = nullptr, *m_texRes = nullptr;
    QDoubleSpinBox *m_scale = nullptr, *m_sharp = nullptr;
    QSpinBox *m_headroom = nullptr;
    QCheckBox *m_situation = nullptr, *m_trace = nullptr,
              *m_killFxaa = nullptr, *m_killMsaa = nullptr, *m_pager = nullptr;
    QPushButton *m_launch = nullptr;
    QPlainTextEdit *m_log = nullptr;
};

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    app.setStyle("Fusion");
    app.setWindowIcon(mvIcon());
    Launcher w;
    w.show();
    return app.exec();
}
