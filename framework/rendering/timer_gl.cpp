#include "timer_gl.hpp"

#include <glbinding/gl/enum.h>
using namespace gl;
#include <globjects/Query.h>

TimerGL::TimerGL()
 :m_query{new globjects::Query}
 ,m_start{0}
 ,m_end{0}
{}

void TimerGL::begin() {
  m_query->counter();
}

void TimerGL::end() {
  // get result from start
  m_start = m_query->get64(GL_QUERY_RESULT);
  m_query->counter();
}

bool TimerGL::outdated() const {
  return m_end < m_start;
}

std::uint64_t TimerGL::duration() const {
  // get end time only if outdated
  if(outdated()) {
    m_end = m_query->waitAndGet64(GL_QUERY_RESULT);
  }
  return m_end - m_start;
}