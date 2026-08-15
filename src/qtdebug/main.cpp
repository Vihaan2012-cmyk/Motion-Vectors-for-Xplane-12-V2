// MOTION VECTORS - DEBUG CONSOLE
//
// A separate application, deliberately, and the reason is the whole point of it:
// it runs BESIDE X-Plane rather than inside it. Every control here takes effect
// on the next frame of a sim that is already flying, and every reading here
// comes from a sim that does not have to be alt-tabbed to.
//
// The problem it solves is not that the layer was hard to observe. It is that
// observing it cost a launch, a look, and a kill - about four minutes - and the
// answer was usually a single number. A dozen of those is an afternoon. Every
// dead end in this project's history has the same shape: a fact that was
// perfectly knowable, that nobody had printed next to the fact it contradicted,
// because printing it would have meant another launch.
//
// Two channels, and neither needs the sim restarted:
//
//   OUT  the live control file, re-read by the layer every few frames. Sliders
//        and toggles here write it; the layer picks the change up next frame.
//
//   IN   the shared memory block the plugin publishes, mapped READ-ONLY here.
//        Camera, jitter, matrices, residual, reset reason, viewport - the same
//        values the in-sim panel shows, without needing the sim in focus.
//
// Plus a tail of the layer log, which is where anything that cannot be a number
// ends up.

#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>
#include <QSlider>
#include <QDoubleSpinBox>
#include <QPlainTextEdit>
#include <QLineEdit>
#include <QTimer>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QSplitter>
#include <QTabWidget>
#include <QStatusBar>
#include <QDir>
#include <QIcon>
#include <QPainter>
#include <QPixmap>
#include <QScrollBar>

#include <windows.h>
#include <cmath>

#ifndef MV_VERSION
#define MV_VERSION "dev"
#endif

// ---------------------------------------------------------------------------
// The shared block, mapped READ-ONLY.
//
// share.h is included rather than redeclared. It is self-contained - stdint,
// string, math and nothing else - so there is no reason to keep a second copy of
// the struct in step with it, and a second copy is exactly the failure the
// VERSION file exists to prevent one directory over.
//
// Read-only is not a detail. This app watches a flying aircraft, and a stray
// write into the block the layer reads its matrices from would be a bug that
// presents as a rendering fault, in the other process, minutes later.
// ---------------------------------------------------------------------------
#include "../share.h"
#include "../mv_icon.h"

struct ShareView {
    HANDLE          handle = nullptr;
    const TaaShare *s = nullptr;

    bool open()
    {
        if (s) return true;
        handle = OpenFileMappingA(FILE_MAP_READ, FALSE, TAA_SHARE_NAME);
        if (!handle) return false;
        s = (const TaaShare *)MapViewOfFile(handle, FILE_MAP_READ, 0, 0, sizeof(TaaShare));
        if (!s) { CloseHandle(handle); handle = nullptr; return false; }
        return true;
    }
    // The block is only trustworthy if it says so itself. A mismatched magic or
    // structSize means the plugin was built from a different share.h, and
    // interpreting it anyway would show plausible nonsense - worse than showing
    // nothing, because nothing is obviously nothing.
    bool sane() const
    {
        return s && s->magic == TAA_MAGIC && s->structSize == sizeof(TaaShare);
    }
};

// ---------------------------------------------------------------------------
// The live control file. Read-modify-write, preserving comments.
//
// It has to preserve comments because the file documents itself - the layer
// writes a fully commented template on first run, and a control panel that
// stripped that would make the file useless to edit by hand, which is the other
// half of how it gets used.
// ---------------------------------------------------------------------------
class LiveFile {
public:
    QString path() const
    {
        if (!m_path.isEmpty()) return m_path;
        QByteArray e = qgetenv("TAA_LIVE_FILE");
        if (!e.isEmpty()) return QString::fromLocal8Bit(e);
        QByteArray t = qgetenv("TEMP");
        return QString::fromLocal8Bit(t.isEmpty() ? "." : t) + "\\taa_live.ini";
    }
    void setPath(const QString &p) { m_path = p; }

    void set(const QString &key, const QString &value)
    {
        QStringList lines;
        QFile f(path());
        if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream in(&f);
            while (!in.atEnd()) lines << in.readLine();
            f.close();
        }
        bool done = false;
        for (int i = 0; i < lines.size(); ++i) {
            QString t = lines[i].trimmed();
            if (t.startsWith('#') || t.startsWith(';')) continue;
            int eq = t.indexOf('=');
            if (eq < 0) continue;
            if (t.left(eq).trimmed() != key) continue;
            lines[i] = key + "=" + value;
            done = true;
            break;
        }
        if (!done) lines << (key + "=" + value);
        QFile o(path());
        if (!o.open(QIODevice::WriteOnly | QIODevice::Text)) return;
        QTextStream out(&o);
        for (const QString &l : lines) out << l << "\n";
    }

    QString get(const QString &key, const QString &dflt = QString()) const
    {
        QFile f(path());
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return dflt;
        QTextStream in(&f);
        QString v = dflt;
        while (!in.atEnd()) {
            QString t = in.readLine().trimmed();
            if (t.startsWith('#') || t.startsWith(';')) continue;
            int eq = t.indexOf('=');
            if (eq < 0) continue;
            if (t.left(eq).trimmed() == key) v = t.mid(eq + 1).trimmed();
        }
        return v;
    }
private:
    QString m_path;
};

// A readable value row that can be coloured. Colour carries meaning here: a
// reading that is merely unusual and a reading that is WRONG look identical in
// a column of numbers, and the whole cost of this project has been in telling
// those two apart.
class Readout : public QLabel {
public:
    explicit Readout(QWidget *p = nullptr) : QLabel(p)
    {
        setTextInteractionFlags(Qt::TextSelectableByMouse);
        QFont f("Consolas"); f.setStyleHint(QFont::Monospace); setFont(f);
    }
    void put(const QString &s, int state = 0)   // 0 normal, 1 good, 2 warn, 3 bad
    {
        setText(s);
        static const char *col[] = { "", "color:#2e7d32;", "color:#e65100;", "color:#c62828;" };
        setStyleSheet(col[qBound(0, state, 3)]);
    }
};

class Console : public QMainWindow {
public:
    Console()
    {
        setWindowTitle(QString("Motion Vectors - Debug Console  %1").arg(MV_VERSION));
        resize(1180, 820);

        auto *tabs = new QTabWidget;
        tabs->addTab(buildControls(), "Controls");
        tabs->addTab(buildTelemetry(), "Telemetry");
        tabs->addTab(buildLog(),       "Layer log");

        setCentralWidget(tabs);
        statusBar()->showMessage("Looking for X-Plane...");

        // Two cadences. Telemetry at 10 Hz because it is a live picture of a
        // flying aircraft; the log at 2 Hz because it is a file tail and reading
        // it faster only burns I/O.
        auto *fast = new QTimer(this);
        connect(fast, &QTimer::timeout, this, &Console::tickTelemetry);
        fast->start(100);
        auto *slow = new QTimer(this);
        connect(slow, &QTimer::timeout, this, &Console::tickLog);
        slow->start(500);

        loadFromFile();
    }

private:
    LiveFile  m_live;
    ShareView m_share;

    // controls
    QCheckBox *m_enable = nullptr;
    QComboBox *m_mode = nullptr, *m_viz = nullptr;
    QDoubleSpinBox *m_alpha = nullptr, *m_gain = nullptr,
                   *m_varclip = nullptr, *m_vizScale = nullptr;
    QCheckBox *m_freeze = nullptr, *m_noMotion = nullptr,
              *m_noAccum = nullptr, *m_forceReset = nullptr;
    QLineEdit *m_livePath = nullptr;
    bool m_loading = false;

    // telemetry
    QMap<QString, Readout*> m_ro;

    // log
    QPlainTextEdit *m_log = nullptr;
    QLineEdit      *m_filter = nullptr;
    QCheckBox      *m_follow = nullptr;
    qint64          m_logPos = 0;
    QString         m_logPath;

    Readout *ro(QFormLayout *form, const char *label)
    {
        auto *r = new Readout;
        form->addRow(label, r);
        m_ro[label] = r;
        return r;
    }

    // ---------------------------------------------------------------- controls
    QWidget *buildControls()
    {
        auto *w = new QWidget;
        auto *v = new QVBoxLayout(w);

        auto *hint = new QLabel(
            "<b>Every control here takes effect on the next frame.</b> "
            "Nothing needs X-Plane restarted. The layer re-reads its control "
            "file a few times a second, so changing a value is as fast as "
            "moving the slider.");
        hint->setWordWrap(true);
        v->addWidget(hint);

        // ---- resolve
        auto *g1 = new QGroupBox("Resolve");
        auto *f1 = new QFormLayout(g1);
        m_enable = new QCheckBox("Enabled  (off leaves the frame completely untouched)");
        f1->addRow(m_enable);

        m_mode = new QComboBox;
        m_mode->addItem("0 - passthrough: bindings and dispatch run, output = input", 0);
        m_mode->addItem("1 - reproject only: ghosting along motion is EXPECTED", 1);
        m_mode->addItem("2 - full: neighbourhood statistics, ghosting collapses", 2);
        f1->addRow("Mode", m_mode);

        auto *modeHelp = new QLabel(
            "<i>Mode 0 is the plumbing test. If the frame changes at all in "
            "passthrough, the fault is the wiring - wrong image bound, bad "
            "barrier, layout mismatch, dispatch size - and not the maths. "
            "Start here whenever anything looks wrong.</i>");
        modeHelp->setWordWrap(true);
        f1->addRow(modeHelp);

        m_alpha   = spin(f1, "Alpha  (current frame weight)", 0.01, 1.0, 0.01, 3);
        m_gain    = spin(f1, "Gain  (how hard rejection pushes alpha to 1)", 0.0, 32.0, 0.25, 2);
        m_varclip = spin(f1, "Variance clip  (box half-width, sigma)", 0.25, 8.0, 0.05, 2);
        v->addWidget(g1);

        // ---- isolation
        auto *g2 = new QGroupBox("Isolate - remove one input at a time");
        auto *f2 = new QVBoxLayout(g2);
        auto *isoHelp = new QLabel(
            "Each of these makes a <b>different prediction</b>, so what you see "
            "attributes the fault instead of just changing the picture.");
        isoHelp->setWordWrap(true);
        f2->addWidget(isoHelp);
        m_freeze = new QCheckBox("Freeze history");
        m_noMotion = new QCheckBox("No motion");
        m_noAccum = new QCheckBox("No accumulation");
        m_forceReset = new QCheckBox("Force reset");
        // QCheckBox cannot wrap its own text, and these labels are long on
        // purpose - each one states the PREDICTION it makes, which is the part
        // that turns a toggle into an experiment. So the sentence goes in a
        // wrapping label beneath a short checkbox.
        struct { QCheckBox *box; const char *say; } iso[] = {
            { m_freeze,     "The image should freeze and smear along motion. If it does "
                            "not, what is on screen is not the history." },
            { m_noMotion,   "Every vector reads zero, so reprojection becomes a same-pixel "
                            "fetch. Ghosting that SURVIVES this is not the vectors." },
            { m_noAccum,    "Current frame out, every binding, barrier and dispatch still "
                            "live." },
            { m_forceReset, "Drops history every frame. Same picture as no-accumulation but "
                            "reached through the reset path, so the pair separates a broken "
                            "reset from broken accumulation." },
        };
        for (auto &e : iso) {
            f2->addWidget(e.box);
            auto *l = new QLabel(QString("<i>%1</i>").arg(e.say));
            l->setWordWrap(true);
            l->setContentsMargins(24, 0, 0, 8);
            f2->addWidget(l);
        }
        v->addWidget(g2);

        // ---- visualise
        auto *g3 = new QGroupBox("Look at it");
        auto *f3 = new QFormLayout(g3);
        m_viz = new QComboBox;
        m_viz->addItem("0 - off, normal image", 0);
        m_viz->addItem("1 - motion vectors as colour (sky and cloud must be FLAT GREY)", 1);
        m_viz->addItem("2 - magnitude heatmap (blue 1px, green 4, yellow 16, red 64)", 2);
        m_viz->addItem("3 - invalid pixels (RED no vector, BLUE off-screen, GREEN ok)", 3);
        m_viz->addItem("4 - history buffer directly", 4);
        m_viz->addItem("5 - blend weight (black accumulate, white current-only)", 5);
        m_viz->addItem("6 - clamp distance in sigma", 6);
        f3->addRow("View", m_viz);
        m_vizScale = spin(f3, "View scale", 0.05, 20.0, 0.05, 2);
        auto *vizHelp = new QLabel(
            "<i>View 3 is the one that can falsify the cloud and sky rejection: "
            "red should cover the sky and the clouds and <b>nothing else</b>. "
            "Red on terrain is a shader we failed to patch.</i>");
        vizHelp->setWordWrap(true);
        f3->addRow(vizHelp);
        v->addWidget(g3);

        // ---- actions
        auto *row = new QHBoxLayout;
        auto *rep = new QPushButton("Write full state to log");
        connect(rep, &QPushButton::clicked, [this]{
            m_live.set("report", "1");
            statusBar()->showMessage("Requested - it appears in the layer log within a frame", 4000);
        });
        auto *reload = new QPushButton("Reload from file");
        connect(reload, &QPushButton::clicked, [this]{ loadFromFile(); });
        auto *open = new QPushButton("Open control file");
        connect(open, &QPushButton::clicked, [this]{
            ShellExecuteA(nullptr, "open", m_live.path().toLocal8Bit().constData(),
                          nullptr, nullptr, SW_SHOW);
        });
        row->addWidget(rep); row->addWidget(reload); row->addWidget(open);
        row->addStretch();
        v->addLayout(row);

        auto *pathRow = new QHBoxLayout;
        pathRow->addWidget(new QLabel("Control file:"));
        m_livePath = new QLineEdit(m_live.path());
        m_livePath->setReadOnly(true);
        pathRow->addWidget(m_livePath);
        v->addLayout(pathRow);

        v->addStretch();

        // Wire everything to write immediately. A control that needs an Apply
        // button is a control you forget to apply, and then you are debugging
        // the button.
        connect(m_enable, &QCheckBox::toggled, [this](bool b){ push("taa.enable", b); });
        connect(m_freeze, &QCheckBox::toggled, [this](bool b){ push("taa.freeze_history", b); });
        connect(m_noMotion, &QCheckBox::toggled, [this](bool b){ push("taa.no_motion", b); });
        connect(m_noAccum, &QCheckBox::toggled, [this](bool b){ push("taa.no_accum", b); });
        connect(m_forceReset, &QCheckBox::toggled, [this](bool b){ push("taa.force_reset", b); });
        connect(m_mode, QOverload<int>::of(&QComboBox::currentIndexChanged),
                [this](int i){ pushI("taa.mode", i); });
        connect(m_viz, QOverload<int>::of(&QComboBox::currentIndexChanged),
                [this](int i){ pushI("taa.viz", i); });
        connect(m_alpha, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                [this](double d){ pushF("taa.alpha", d); });
        connect(m_gain, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                [this](double d){ pushF("taa.gain", d); });
        connect(m_varclip, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                [this](double d){ pushF("taa.varclip", d); });
        connect(m_vizScale, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                [this](double d){ pushF("taa.viz_scale", d); });
        return w;
    }

    QDoubleSpinBox *spin(QFormLayout *f, const char *label,
                         double lo, double hi, double step, int dec)
    {
        auto *s = new QDoubleSpinBox;
        s->setRange(lo, hi); s->setSingleStep(step); s->setDecimals(dec);
        f->addRow(label, s);
        return s;
    }

    void push(const QString &k, bool b)   { if (!m_loading) m_live.set(k, b ? "1" : "0"); }
    void pushI(const QString &k, int i)   { if (!m_loading) m_live.set(k, QString::number(i)); }
    void pushF(const QString &k, double d){ if (!m_loading) m_live.set(k, QString::number(d, 'g', 6)); }

    void loadFromFile()
    {
        m_loading = true;
        m_enable->setChecked(m_live.get("taa.enable", "0") != "0");
        m_mode->setCurrentIndex(qBound(0, m_live.get("taa.mode", "0").toInt(), 2));
        m_viz->setCurrentIndex(qBound(0, m_live.get("taa.viz", "0").toInt(), 6));
        m_alpha->setValue(m_live.get("taa.alpha", "0.1").toDouble());
        m_gain->setValue(m_live.get("taa.gain", "4").toDouble());
        m_varclip->setValue(m_live.get("taa.varclip", "1.25").toDouble());
        m_vizScale->setValue(m_live.get("taa.viz_scale", "1").toDouble());
        m_freeze->setChecked(m_live.get("taa.freeze_history", "0") != "0");
        m_noMotion->setChecked(m_live.get("taa.no_motion", "0") != "0");
        m_noAccum->setChecked(m_live.get("taa.no_accum", "0") != "0");
        m_forceReset->setChecked(m_live.get("taa.force_reset", "0") != "0");
        m_livePath->setText(m_live.path());
        m_loading = false;
    }

    // --------------------------------------------------------------- telemetry
    QWidget *buildTelemetry()
    {
        auto *w = new QWidget;
        auto *outer = new QVBoxLayout(w);
        auto *note = new QLabel(
            "Read live from the plugin's shared memory block, read-only. These "
            "are the same values the in-sim panel shows, without needing the sim "
            "in focus - so they can be watched while flying.");
        note->setWordWrap(true);
        outer->addWidget(note);

        auto *cols = new QHBoxLayout;

        auto *gA = new QGroupBox("Link");
        auto *fA = new QFormLayout(gA);
        ro(fA, "Shared block");
        ro(fA, "Frame");
        ro(fA, "Struct size");
        ro(fA, "Paused");
        ro(fA, "View type");
        cols->addWidget(gA);

        auto *gB = new QGroupBox("Camera");
        auto *fB = new QFormLayout(gB);
        ro(fB, "Position");
        ro(fB, "Delta (m/frame)");
        ro(fB, "Viewport");
        ro(fB, "FOV");
        ro(fB, "Near / far");
        cols->addWidget(gB);

        auto *gC = new QGroupBox("Temporal");
        auto *fC = new QFormLayout(gC);
        ro(fC, "Jitter");
        ro(fC, "Jitter phase");
        ro(fC, "Reproj valid");
        ro(fC, "History reset");
        ro(fC, "Render scale");
        cols->addWidget(gC);

        outer->addLayout(cols);
        outer->addStretch();
        return w;
    }

    void tickTelemetry()
    {
        if (!m_share.open()) {
            m_ro["Shared block"]->put("not found - X-Plane is not running, or the "
                                      "plugin has not loaded yet", 2);
            statusBar()->showMessage("X-Plane not detected");
            for (auto it = m_ro.begin(); it != m_ro.end(); ++it)
                if (it.key() != "Shared block") it.value()->put("-");
            return;
        }
        if (!m_share.sane()) {
            m_ro["Shared block"]->put(
                QString("mapped, but magic/size disagree (want %1/%2) - the plugin "
                        "was built from a different share.h. Refusing to interpret it.")
                    .arg(TAA_MAGIC, 0, 16).arg((int)sizeof(TaaShare)), 3);
            return;
        }
        const TaaShare &t = *m_share.s;
        m_ro["Shared block"]->put(QString("connected   version %1").arg(t.version), 1);
        m_ro["Struct size"]->put(QString::number(t.structSize));
        statusBar()->showMessage(QString("Connected to X-Plane   frame %1").arg(t.frame));

        m_ro["Frame"]->put(QString::number((qulonglong)t.frame));
        m_ro["Paused"]->put(t.paused ? "yes" : "no", t.paused ? 2 : 0);
        m_ro["View type"]->put(QString::number(t.viewType));
        m_ro["Position"]->put(QString("%1 %2 %3")
            .arg(t.camX, 11, 'f', 1).arg(t.camY, 11, 'f', 1).arg(t.camZ, 11, 'f', 1));
        m_ro["Delta (m/frame)"]->put(QString::number(t.camDelta, 'f', 5));
        m_ro["Viewport"]->put(QString("%1 x %2").arg(t.viewportW).arg(t.viewportH));
        m_ro["FOV"]->put(QString::number(t.fovDeg, 'f', 2));
        m_ro["Near / far"]->put(QString("%1 / %2%3")
            .arg(t.nearClip, 0, 'f', 4)
            .arg(t.farClip, 0, 'f', 1)
            .arg(t.infiniteFar ? "  (infinite)" : ""));
        m_ro["Jitter"]->put(QString("%1 %2")
            .arg(t.jitterX, 10, 'f', 5).arg(t.jitterY, 10, 'f', 5));
        m_ro["Jitter phase"]->put(QString("%1 of %2")
            .arg(t.jitterIndex).arg(t.jitterPhases));
        m_ro["Reproj valid"]->put(t.reprojValid ? "yes"
                                                : "NO - currViewProj was singular",
                                  t.reprojValid ? 1 : 3);
        m_ro["History reset"]->put(t.historyReset
                                     ? QString("YES   reason %1").arg(t.resetReason)
                                     : "no",
                                   t.historyReset ? 2 : 0);
        m_ro["Render scale"]->put(QString::number(t.renderScale, 'f', 4));
    }

    // --------------------------------------------------------------------- log
    QWidget *buildLog()
    {
        auto *w = new QWidget;
        auto *v = new QVBoxLayout(w);

        auto *row = new QHBoxLayout;
        row->addWidget(new QLabel("Filter:"));
        m_filter = new QLineEdit;
        m_filter->setPlaceholderText("substring, case-insensitive - try TAA, DECLINED, "
                                     "MISMATCH, LIVE, GBUFFER_VEL");
        row->addWidget(m_filter);
        m_follow = new QCheckBox("Follow"); m_follow->setChecked(true);
        row->addWidget(m_follow);
        auto *clear = new QPushButton("Clear view");
        connect(clear, &QPushButton::clicked, [this]{ m_log->clear(); });
        row->addWidget(clear);
        // Quick filters for the things that are always the question.
        for (const char *k : { "DECLINED", "MISMATCH", "FEEDBACK", "FULL STATE" }) {
            auto *b = new QPushButton(k);
            QString kk(k);
            connect(b, &QPushButton::clicked, [this, kk]{ m_filter->setText(kk); });
            row->addWidget(b);
        }
        v->addLayout(row);

        m_log = new QPlainTextEdit;
        m_log->setReadOnly(true);
        m_log->setMaximumBlockCount(20000);
        QFont f("Consolas"); f.setStyleHint(QFont::Monospace); m_log->setFont(f);
        v->addWidget(m_log);

        QByteArray t = qgetenv("TEMP");
        m_logPath = QString::fromLocal8Bit(t.isEmpty() ? "." : t) + "\\taa_layer.txt";
        auto *lp = new QLabel("Tailing " + m_logPath +
                              "   (the layer writes this only when TAA_LAYER_TRACE is set)");
        lp->setTextInteractionFlags(Qt::TextSelectableByMouse);
        v->addWidget(lp);
        return w;
    }

    void tickLog()
    {
        QFile f(m_logPath);
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;
        // A shrinking file means the sim restarted and the log was truncated.
        // Start over rather than seeking past the end and showing nothing, which
        // reads exactly like "the layer stopped logging".
        if (f.size() < m_logPos) { m_logPos = 0; m_log->clear(); }
        if (!f.seek(m_logPos)) return;
        QByteArray chunk = f.read(1 << 20);
        m_logPos = f.pos();
        if (chunk.isEmpty()) return;

        const QString filter = m_filter->text().trimmed();
        const QStringList lines = QString::fromLocal8Bit(chunk).split('\n');
        for (const QString &l : lines) {
            if (l.isEmpty()) continue;
            if (!filter.isEmpty() && !l.contains(filter, Qt::CaseInsensitive)) continue;
            m_log->appendPlainText(l);
        }
        if (m_follow->isChecked())
            m_log->verticalScrollBar()->setValue(m_log->verticalScrollBar()->maximum());
    }
};

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    app.setApplicationName("Motion Vectors Debug Console");
    app.setWindowIcon(mvIcon());
    Console c;
    c.setWindowIcon(mvIcon());
    c.show();
    return app.exec();
}
