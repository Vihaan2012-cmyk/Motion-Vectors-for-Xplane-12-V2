#pragma once

// ONE LOGO, DRAWN, SHARED BY BOTH APPLICATIONS.
//
// Drawn rather than shipped as a .ico for two reasons. There is no binary asset
// to keep in step with anything - the same failure the VERSION file exists to
// prevent, and the same failure that let a stale launcher ship for months. And
// both applications call the same function, so they cannot drift apart: a logo
// that is nearly the same in two places is worse than no logo, because it reads
// as two different products.
//
// The mark is a velocity field: four arrows, lengthening and shifting hue as
// they go, which is what the whole project is about. It scales down to 16px
// without turning to mud, which a more detailed mark would.

#include <QIcon>
#include <QPixmap>
#include <QPainter>
#include <QColor>

inline QIcon mvIcon()
{
    // Rendered once. A QIcon built per call would repaint on every window that
    // asks for it, and both apps ask several times.
    static QIcon cached;
    if (!cached.isNull()) return cached;

    QPixmap pm(256, 256);
    pm.fill(Qt::transparent);
    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing);

    // Rounded dark plate, so the light arrows read on any taskbar theme.
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(22, 28, 42));
    p.drawRoundedRect(8, 8, 240, 240, 48, 48);

    for (int i = 0; i < 4; ++i) {
        const qreal y   = 66 + i * 42;
        const qreal len = 52 + i * 32;      // longer toward the bottom, as a
                                            // real field is nearer the camera
        const qreal x0  = 48;
        const qreal x1  = x0 + len;
        const QColor c  = QColor::fromHsv(205 - i * 42, 210, 250);

        QPen pen(c, 14, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        p.setPen(pen);
        p.drawLine(QPointF(x0, y), QPointF(x1, y));
        // Arrow head, drawn as two strokes so the round caps do the shaping.
        p.drawLine(QPointF(x1, y), QPointF(x1 - 22, y - 17));
        p.drawLine(QPointF(x1, y), QPointF(x1 - 22, y + 17));
    }
    p.end();

    cached = QIcon(pm);
    return cached;
}
