#include <QGuiApplication>
#include <QCommandLineParser>
#include <QTextStream>
#include <QPainter>
#include <QImageWriter>
#include <QJsonDocument>
#include <QJsonArray>
#include <QDebug>
#include <algorithm>

static QTextStream cin(stdin);
static QTextStream cout(stdout);
static QTextStream cerr(stderr);

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    cin.setCodec("utf8");

    QCommandLineParser parser;
    parser.process(app);
    const QStringList args = parser.positionalArguments();
    if (args.size() < 2) {
        cerr << "missing parameters" << endl;
        return 1;
    }

    QImage img(args[0]);
    QColor color(args[2]);
    QJsonDocument doc = QJsonDocument::fromJson(cin.readLine().toUtf8());

    QVector<QPair<QRectF, QString> > v;
    QJsonArray array = doc.array();
    for (int i = 0; i < array.size(); i++) {
        QString text = array[i].toObject()["text"].toString();
        QJsonArray bbox = array[i].toObject()["bbox"].toArray();
        double xmin = bbox[0].toDouble();
        double ymin = bbox[1].toDouble();
        double w = bbox[2].toDouble();
        double h = bbox[3].toDouble();
        QRectF rect(xmin, ymin, w, h);
        v.push_back(qMakePair(rect, text));
    }
    std::sort(v.begin(), v.end(), [](QPair<QRectF, QString> const &a, QPair<QRectF, QString> const &b) {
        return a.first.topLeft().x() == b.first.topLeft().x() ?
                    a.first.topLeft().y() < b.first.topLeft().y() :
                    a.first.topLeft().x() < b.first.topLeft().x();
    });
    QPainter painter(&img);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
    QVector<QRectF> forbidden_area;
    for (int i = v.size() - 1; i >= 0; i--) {
        QRectF bbox = v[i].first;
        forbidden_area.append(bbox);
    }
    for (int i = v.size() - 1; i >= 0; i--) {
        QRectF bbox = v[i].first;
        QString text = v[i].second;
        if (text.isEmpty())
            text = app.tr("■");
        QPen pen(color);
        pen.setWidth(2);
        painter.setOpacity(1.0);
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(bbox);
        double higher = 8.0;
        int fontsize = 16.0;
        double lefter = 6.0;
        double deltax = 2.0;
        forbidden_area.append(bbox);
        double texty = bbox.topLeft().y() - higher;
        double textx0 = bbox.topLeft().x() - lefter;
        if (textx0 < fontsize)
            textx0 = (double)fontsize;
        double textx = textx0;
        QRectF textArea(0, 0, 0, 0);
        auto isForbidden = [&]() {
            textArea = QRectF(textx - fontsize, texty - fontsize, fontsize, fontsize);
            foreach (QRectF const &r, forbidden_area) {
                QRectF inter = textArea.intersected(r);
                if (inter.isValid())
                    return true;
            }
            return false;
        };
        while (isForbidden()) {
            if (textx > fontsize) {
                textx = qMax((double)fontsize, textx - deltax);
            } else {
                texty -= deltax;
                if (texty <= fontsize) {
                    texty = qMax(0., texty);
                    break;
                }
            }
        }
        pen.setWidth(1);
        painter.setPen(pen);
        painter.setOpacity(0.5);
        painter.drawLine(bbox.topLeft(), QPointF(textx0, texty));
        painter.drawLine(QPointF(textx - fontsize, texty), QPointF(textx0, texty));
        painter.setOpacity(1.0);
        painter.drawText(textArea, Qt::AlignCenter, text);
        forbidden_area.append(textArea);
    }
    painter.end();
    QImageWriter imageWriter;
    imageWriter.setFileName(args[1]);
    imageWriter.setQuality(40);
    imageWriter.write(img);
    return 0;
}
