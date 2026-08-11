// Qt launcher for Motion Vectors.
//
// Its job is small and it should stay small: set the two Vulkan loader
// variables and start X-Plane. Everything else here is reporting - what it
// found, and whether a newer build exists.
//
// The variables are what make the layer EXPLICIT. Registering it through the
// loader's ImplicitLayers key would load the DLL into every Vulkan application
// on the machine; setting these here scopes it to the process this launcher
// starts, and nothing else on the system is touched.

#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QDir>
#include <QFileInfo>
#include <QFileDialog>
#include <QSettings>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QJsonDocument>
#include <QJsonObject>
#include <QDesktopServices>
#include <QUrl>

static const char *kVersion  = "0.0.06";
static const char *kLayer    = "VK_LAYER_mv";
static const char *kReleases =
    "https://api.github.com/repos/Vihaan2012-cmyk/Motion-Vectors-for-Xplane-12-V2/releases/latest";

class Launcher : public QWidget {
public:
    Launcher()
    {
        setWindowTitle(QString("Motion Vectors  %1").arg(kVersion));
        setMinimumWidth(560);

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

        m_situation = new QCheckBox("Start straight into the saved situation (debugging)");
        m_situation->setToolTip(
            "Uses --load_smo. It is the only option that bypasses the flight "
            "configuration screen: --load_acf, --load_apt, --go_to_apt and "
            "--new_flight_json are all parsed but only pre-fill it.");
        root->addWidget(m_situation);

        m_trace = new QCheckBox("Write the layer log (slower - it logs from the render path)");
        root->addWidget(m_trace);

        m_launch = new QPushButton("Launch X-Plane 12");
        m_launch->setMinimumHeight(38);
        connect(m_launch, &QPushButton::clicked, this, &Launcher::launch);
        root->addWidget(m_launch);

        m_update = new QLabel;
        m_update->setOpenExternalLinks(true);
        m_update->setWordWrap(true);
        root->addWidget(m_update);

        m_log = new QPlainTextEdit;
        m_log->setReadOnly(true);
        m_log->setMaximumHeight(90);
        root->addWidget(m_log);

        QSettings s("MotionVectors", "Launcher");
        m_root = s.value("xplane").toString();
        if (m_root.isEmpty() || !hasXPlane(m_root)) m_root = guessRoot();

        refresh();
        checkForUpdate();
    }

private:
    static bool hasXPlane(const QString &dir)
    {
        return QFileInfo::exists(QDir(dir).filePath("X-Plane.exe"));
    }

    // The launcher is installed inside the X-Plane folder, so look at our own
    // location first and only then at the usual places.
    QString guessRoot() const
    {
        QDir here(QCoreApplication::applicationDirPath());
        if (hasXPlane(here.absolutePath())) return here.absolutePath();
        here.cdUp();
        if (hasXPlane(here.absolutePath())) return here.absolutePath();
        const QStringList guesses = {
            "D:/Steam Games/steamapps/common/X-Plane 12",
            "C:/Program Files/X-Plane 12",
            "C:/X-Plane 12",
        };
        for (const QString &g : guesses)
            if (hasXPlane(g)) return g;
        return QString();
    }

    QString layerDir() const
    {
        // Beside us when installed; under build/ when run from the tree.
        QDir d(QCoreApplication::applicationDirPath());
        if (QFileInfo::exists(d.filePath("VkLayer_mv.json"))) return d.absolutePath();
        QDir b(QCoreApplication::applicationDirPath());
        b.cd("build"); b.cd("vklayer");
        return b.absolutePath();
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
            s = "<b style='color:#c0392b'>Layer not found.</b> The manifest VkLayer_mv.json is missing - reinstall.";
        else
            s = "<b style='color:#1e8449'>Ready.</b> The layer is enabled for X-Plane only, "
                "so no other Vulkan application loads it.";
        m_status->setText(s);
        m_path->setText(okXP ? QString("X-Plane: %1<br>Layer: %2").arg(m_root, layerDir())
                             : QString("Layer: %1").arg(layerDir()));
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

    void launch()
    {
        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        env.insert("VK_LAYER_PATH", QDir::toNativeSeparators(layerDir()));
        env.insert("VK_LOADER_LAYERS_ENABLE", kLayer);
        if (m_trace->isChecked()) env.insert("TAA_LAYER_TRACE", "1");

        QStringList args;
        if (m_situation->isChecked())
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
        m_log->appendPlainText("Started with " + QString(kLayer) +
                               (args.isEmpty() ? "" : "  " + args.join(' ')));
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
    QLabel *m_status, *m_path, *m_update;
    QCheckBox *m_situation, *m_trace;
    QPushButton *m_launch;
    QPlainTextEdit *m_log;
};

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    app.setStyle("Fusion");
    Launcher w;
    w.show();
    return app.exec();
}
