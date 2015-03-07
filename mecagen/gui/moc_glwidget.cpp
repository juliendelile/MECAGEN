/****************************************************************************
** Meta object code from reading C++ file 'glwidget.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.2.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "includes/glwidget.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'glwidget.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.2.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_mg__GLWidget_t {
    QByteArrayData data[24];
    char stringdata[290];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_mg__GLWidget_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_mg__GLWidget_t qt_meta_stringdata_mg__GLWidget = {
    {
QT_MOC_LITERAL(0, 0, 12),
QT_MOC_LITERAL(1, 13, 16),
QT_MOC_LITERAL(2, 30, 0),
QT_MOC_LITERAL(3, 31, 5),
QT_MOC_LITERAL(4, 37, 16),
QT_MOC_LITERAL(5, 54, 16),
QT_MOC_LITERAL(6, 71, 12),
QT_MOC_LITERAL(7, 84, 12),
QT_MOC_LITERAL(8, 97, 12),
QT_MOC_LITERAL(9, 110, 13),
QT_MOC_LITERAL(10, 124, 1),
QT_MOC_LITERAL(11, 126, 13),
QT_MOC_LITERAL(12, 140, 13),
QT_MOC_LITERAL(13, 154, 13),
QT_MOC_LITERAL(14, 168, 13),
QT_MOC_LITERAL(15, 182, 13),
QT_MOC_LITERAL(16, 196, 12),
QT_MOC_LITERAL(17, 209, 12),
QT_MOC_LITERAL(18, 222, 9),
QT_MOC_LITERAL(19, 232, 2),
QT_MOC_LITERAL(20, 235, 12),
QT_MOC_LITERAL(21, 248, 18),
QT_MOC_LITERAL(22, 267, 11),
QT_MOC_LITERAL(23, 279, 9)
    },
    "mg::GLWidget\0xRotationChanged\0\0angle\0"
    "yRotationChanged\0zRotationChanged\0"
    "setXRotation\0setYRotation\0setZRotation\0"
    "setSlicerXmax\0i\0setSlicerXmin\0"
    "setSlicerYmax\0setSlicerYmin\0setSlicerZmax\0"
    "setSlicerZmin\0setOrthoProj\0setPerspProj\0"
    "set2Dinfo\0on\0setColByType\0newDisplayParamVBO\0"
    "setDrawAxes\0setAxesId\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_mg__GLWidget[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      19,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    1,  109,    2, 0x06,
       4,    1,  112,    2, 0x06,
       5,    1,  115,    2, 0x06,

 // slots: name, argc, parameters, tag, flags
       6,    1,  118,    2, 0x0a,
       7,    1,  121,    2, 0x0a,
       8,    1,  124,    2, 0x0a,
       9,    1,  127,    2, 0x0a,
      11,    1,  130,    2, 0x0a,
      12,    1,  133,    2, 0x0a,
      13,    1,  136,    2, 0x0a,
      14,    1,  139,    2, 0x0a,
      15,    1,  142,    2, 0x0a,
      16,    0,  145,    2, 0x0a,
      17,    0,  146,    2, 0x0a,
      18,    1,  147,    2, 0x0a,
      20,    1,  150,    2, 0x0a,
      21,    0,  153,    2, 0x0a,
      22,    1,  154,    2, 0x0a,
      23,    1,  157,    2, 0x0a,

 // signals: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   19,
    QMetaType::Void, QMetaType::Bool,   19,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   19,
    QMetaType::Void, QMetaType::Int,   10,

       0        // eod
};

void mg::GLWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        GLWidget *_t = static_cast<GLWidget *>(_o);
        switch (_id) {
        case 0: _t->xRotationChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->yRotationChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->zRotationChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->setXRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->setYRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->setZRotation((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->setSlicerXmax((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->setSlicerXmin((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->setSlicerYmax((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->setSlicerYmin((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: _t->setSlicerZmax((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: _t->setSlicerZmin((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 12: _t->setOrthoProj(); break;
        case 13: _t->setPerspProj(); break;
        case 14: _t->set2Dinfo((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 15: _t->setColByType((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 16: _t->newDisplayParamVBO(); break;
        case 17: _t->setDrawAxes((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 18: _t->setAxesId((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (GLWidget::*_t)(int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GLWidget::xRotationChanged)) {
                *result = 0;
            }
        }
        {
            typedef void (GLWidget::*_t)(int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GLWidget::yRotationChanged)) {
                *result = 1;
            }
        }
        {
            typedef void (GLWidget::*_t)(int );
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&GLWidget::zRotationChanged)) {
                *result = 2;
            }
        }
    }
}

const QMetaObject mg::GLWidget::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_mg__GLWidget.data,
      qt_meta_data_mg__GLWidget,  qt_static_metacall, 0, 0}
};


const QMetaObject *mg::GLWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *mg::GLWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_mg__GLWidget.stringdata))
        return static_cast<void*>(const_cast< GLWidget*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int mg::GLWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 19)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 19;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 19)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 19;
    }
    return _id;
}

// SIGNAL 0
void mg::GLWidget::xRotationChanged(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void mg::GLWidget::yRotationChanged(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void mg::GLWidget::zRotationChanged(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
QT_END_MOC_NAMESPACE
