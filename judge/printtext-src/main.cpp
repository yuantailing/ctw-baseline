#include <QGuiApplication>
#include <QCommandLineParser>
#include <QTextStream>
#include <QPainter>
#include <QImageWriter>
#include <QJsonDocument>
#include <QJsonArray>
#include <QDebug>
#include <algorithm>
#include <cassert>

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
    QJsonDocument doc = QJsonDocument::fromJson(cin.readLine().toUtf8());

    QVector<QPair<QRectF, QPair<QString, QString> > > v;
    QJsonObject object = doc.object();
    QJsonArray array = object["boxes"].toArray();
    QJsonArray jcrop = object["crop"].toArray();
    QRect crop(jcrop[0].toInt(), jcrop[1].toInt(), jcrop[2].toInt(), jcrop[3].toInt());
    QString place = object["place"].toString();
    img = img.copy(crop);
    assert(place == "smart" || place == "force");
    for (int i = 0; i < array.size(); i++) {
        QString text = array[i].toObject()["text"].toString();
        QString color = array[i].toObject()["color"].toString();
        QJsonArray bbox = array[i].toObject()["bbox"].toArray();
        double xmin = bbox[0].toDouble() - crop.x();
        double ymin = bbox[1].toDouble() - crop.y();
        double w = bbox[2].toDouble();
        double h = bbox[3].toDouble();
        QRectF rect(xmin, ymin, w, h);
        v.push_back(qMakePair(rect, qMakePair(text, color)));
    }
    std::sort(v.begin(), v.end(), [](QPair<QRectF, QPair<QString, QString> > const &a, QPair<QRectF, QPair<QString, QString> > const &b) {
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
        QString color = v[i].second.second;
        QPen pen(color);
        pen.setWidth(2);
        painter.setOpacity(1.0);
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(bbox);
    }
    for (int i = v.size() - 1; i >= 0; i--) {
        QRectF bbox = v[i].first;
        QString text = v[i].second.first;
        QString color = v[i].second.second;
        if (text.isEmpty())
            continue;
        double higher = 8.0;
        int fontsize = 22.0;
        int fontpadding = 2.0;
        double lefter = 6.0;
        double deltax = 2.0;
        double texty = bbox.topLeft().y() - higher;
        double textx0 = bbox.topLeft().x() - lefter;
        if (textx0 < fontsize && place != "force")
            textx0 = (double)fontsize;
        double textx = textx0;
        auto textArea = [&]()->QRectF {
            return QRectF(textx - fontsize, texty - fontsize, fontsize, fontsize);
        };
        auto isForbidden = [&]() {
            if (place == "force")
                return false;
            foreach (QRectF const &r, forbidden_area) {
                QRectF inter = textArea().intersected(r);
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
        QPen pen(color);
        pen.setWidth(1);
        painter.setOpacity(0.5);
        painter.setPen(pen);
        painter.setFont(QFont("SimHei", fontsize - fontpadding * 2));
        painter.drawLine(bbox.topLeft(), QPointF(textx0, texty));
        painter.drawLine(QPointF(textx - fontsize, texty), QPointF(textx0, texty));
        painter.setOpacity(1.0);
        painter.drawText(textArea(), Qt::AlignCenter, text);
        forbidden_area.append(textArea());
    }
    painter.end();
    QImageWriter imageWriter;
    imageWriter.setFileName(args[1]);
    imageWriter.setQuality(40);
    imageWriter.write(img);
    return 0;
}
