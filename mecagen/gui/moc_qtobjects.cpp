/****************************************************************************
** Meta object code from reading C++ file 'qtobjects.hpp'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.2.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "includes/qtobjects.hpp"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'qtobjects.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.2.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_mg__ColorLabel_t {
    QByteArrayData data[3];
    char stringdata[31];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_mg__ColorLabel_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_mg__ColorLabel_t qt_meta_stringdata_mg__ColorLabel = {
    {
QT_MOC_LITERAL(0, 0, 14),
QT_MOC_LITERAL(1, 15, 13),
QT_MOC_LITERAL(2, 29, 0)
    },
    "mg::ColorLabel\0doubleClicked\0\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_mg__ColorLabel[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x06,

 // signals: parameters
    QMetaType::Void,

       0        // eod
};

void mg::ColorLabel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ColorLabel *_t = static_cast<ColorLabel *>(_o);
        switch (_id) {
        case 0: _t->doubleClicked(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (ColorLabel::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&ColorLabel::doubleClicked)) {
                *result = 0;
            }
        }
    }
    Q_UNUSED(_a);
}

const QMetaObject mg::ColorLabel::staticMetaObject = {
    { &QLabel::staticMetaObject, qt_meta_stringdata_mg__ColorLabel.data,
      qt_meta_data_mg__ColorLabel,  qt_static_metacall, 0, 0}
};


const QMetaObject *mg::ColorLabel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *mg::ColorLabel::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_mg__ColorLabel.stringdata))
        return static_cast<void*>(const_cast< ColorLabel*>(this));
    return QLabel::qt_metacast(_clname);
}

int mg::ColorLabel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QLabel::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void mg::ColorLabel::doubleClicked()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
struct qt_meta_stringdata_mg__ProtLigRadioButton_t {
    QByteArrayData data[3];
    char stringdata[33];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_mg__ProtLigRadioButton_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_mg__ProtLigRadioButton_t qt_meta_stringdata_mg__ProtLigRadioButton = {
    {
QT_MOC_LITERAL(0, 0, 22),
QT_MOC_LITERAL(1, 23, 7),
QT_MOC_LITERAL(2, 31, 0)
    },
    "mg::ProtLigRadioButton\0clicked\0\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_mg__ProtLigRadioButton[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x0a,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void mg::ProtLigRadioButton::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ProtLigRadioButton *_t = static_cast<ProtLigRadioButton *>(_o);
        switch (_id) {
        case 0: _t->clicked(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObject mg::ProtLigRadioButton::staticMetaObject = {
    { &QRadioButton::staticMetaObject, qt_meta_stringdata_mg__ProtLigRadioButton.data,
      qt_meta_data_mg__ProtLigRadioButton,  qt_static_metacall, 0, 0}
};


const QMetaObject *mg::ProtLigRadioButton::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *mg::ProtLigRadioButton::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_mg__ProtLigRadioButton.stringdata))
        return static_cast<void*>(const_cast< ProtLigRadioButton*>(this));
    return QRadioButton::qt_metacast(_clname);
}

int mg::ProtLigRadioButton::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QRadioButton::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}
struct qt_meta_stringdata_mg__WheeledValueLabel_t {
    QByteArrayData data[3];
    char stringdata[36];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_mg__WheeledValueLabel_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_mg__WheeledValueLabel_t qt_meta_stringdata_mg__WheeledValueLabel = {
    {
QT_MOC_LITERAL(0, 0, 21),
QT_MOC_LITERAL(1, 22, 11),
QT_MOC_LITERAL(2, 34, 0)
    },
    "mg::WheeledValueLabel\0wheelSignal\0\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_mg__WheeledValueLabel[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x06,

 // signals: parameters
    QMetaType::Void,

       0        // eod
};

void mg::WheeledValueLabel::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        WheeledValueLabel *_t = static_cast<WheeledValueLabel *>(_o);
        switch (_id) {
        case 0: _t->wheelSignal(); break;
        default: ;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (WheeledValueLabel::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&WheeledValueLabel::wheelSignal)) {
                *result = 0;
            }
        }
    }
    Q_UNUSED(_a);
}

const QMetaObject mg::WheeledValueLabel::staticMetaObject = {
    { &QLabel::staticMetaObject, qt_meta_stringdata_mg__WheeledValueLabel.data,
      qt_meta_data_mg__WheeledValueLabel,  qt_static_metacall, 0, 0}
};


const QMetaObject *mg::WheeledValueLabel::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *mg::WheeledValueLabel::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_mg__WheeledValueLabel.stringdata))
        return static_cast<void*>(const_cast< WheeledValueLabel*>(this));
    return QLabel::qt_metacast(_clname);
}

int mg::WheeledValueLabel::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QLabel::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void mg::WheeledValueLabel::wheelSignal()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE
