/****************************************************************************
** Meta object code from reading C++ file 'window.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.2.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "includes/window.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'window.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.2.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_mg__Window_t {
    QByteArrayData data[11];
    char stringdata[136];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_mg__Window_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_mg__Window_t qt_meta_stringdata_mg__Window = {
    {
QT_MOC_LITERAL(0, 0, 10),
QT_MOC_LITERAL(1, 11, 9),
QT_MOC_LITERAL(2, 21, 0),
QT_MOC_LITERAL(3, 22, 9),
QT_MOC_LITERAL(4, 32, 11),
QT_MOC_LITERAL(5, 44, 22),
QT_MOC_LITERAL(6, 67, 19),
QT_MOC_LITERAL(7, 87, 17),
QT_MOC_LITERAL(8, 105, 14),
QT_MOC_LITERAL(9, 120, 10),
QT_MOC_LITERAL(10, 131, 3)
    },
    "mg::Window\0callStart\0\0callPause\0"
    "callPlusOne\0showHideDisplayControl\0"
    "showHideCustomParam\0showHideRecording\0"
    "saveStateParam\0setDamping\0val\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_mg__Window[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   54,    2, 0x0a,
       3,    0,   55,    2, 0x0a,
       4,    0,   56,    2, 0x0a,
       5,    0,   57,    2, 0x0a,
       6,    0,   58,    2, 0x0a,
       7,    0,   59,    2, 0x0a,
       8,    0,   60,    2, 0x0a,
       9,    1,   61,    2, 0x0a,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   10,

       0        // eod
};

void mg::Window::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Window *_t = static_cast<Window *>(_o);
        switch (_id) {
        case 0: _t->callStart(); break;
        case 1: _t->callPause(); break;
        case 2: _t->callPlusOne(); break;
        case 3: _t->showHideDisplayControl(); break;
        case 4: _t->showHideCustomParam(); break;
        case 5: _t->showHideRecording(); break;
        case 6: _t->saveStateParam(); break;
        case 7: _t->setDamping((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject mg::Window::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_mg__Window.data,
      qt_meta_data_mg__Window,  qt_static_metacall, 0, 0}
};


const QMetaObject *mg::Window::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *mg::Window::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_mg__Window.stringdata))
        return static_cast<void*>(const_cast< Window*>(this));
    return QWidget::qt_metacast(_clname);
}

int mg::Window::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 8;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
