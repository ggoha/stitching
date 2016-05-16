TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG += c++11

SOURCES += main.cpp

LIBS += `pkg-config opencv --libs`

