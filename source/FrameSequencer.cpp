#include "FrameSequencer.h"
#include <iostream>
#include <vector>
int FrameSequencer::next_frame()
{
    if(is_finished())
    {
        return _frame_end;
    }

    if(_type == Type::FULL_SEQUENCE)
    {
        if(_frame < _frame_end)
        {
            _frame++;
        }

        return _frame;
    }

    if(_type == Type::FULL_SEQUENCE_REPEAT)
    {
        if(_forward_motion)
        {
            _frame++;
        }
        else
        {
            _frame--;
        }

        if(_frame == _frame_end)
        {
            _forward_motion = false;
        }

        return _frame;
    }

    if(_type == Type::INCREASING_STEP)
    {
        _frame = *_precomp_sequence_iterator;

        _precomp_sequence_iterator++;

        // std::cout << _frame << std::endl;
        return _frame;
    }

    return _frame_end;
}
FrameSequencer::FrameSequencer(FrameSequencer::Type type, int frame_start, int frame_end)
{
    _type = type;
    _frame_start = frame_start;
    _frame_end = frame_end;
    _frame = _frame_start;
    _forward_motion = true;

    if(type == FrameSequencer::Type::INCREASING_STEP)
    {
        for(int i = 0; pow(2., (double)i) <= (frame_end - frame_start); i++)
        {
            _precomp_sequence.emplace_back(frame_start);
            _precomp_sequence.emplace_back(frame_start + pow(2., (double)i));
        }

        _precomp_sequence.emplace_back(frame_start);
        _precomp_sequence.emplace_back(frame_end);
    }

    _precomp_sequence_iterator = _precomp_sequence.begin();
}
bool FrameSequencer::is_finished() { return ((_frame == _frame_end) && _forward_motion) || ((_frame == _frame_start) && !_forward_motion); }
bool FrameSequencer::is_first_frame() { return _frame == _frame_start && _forward_motion; }
int FrameSequencer::length() { return _frame_end - _frame_start; }
void FrameSequencer::rewind()
{
    _frame = _frame_start;
    _forward_motion = true;
    _precomp_sequence_iterator = _precomp_sequence.begin();
}
int FrameSequencer::next_frame_position()
{
    if(_type == FrameSequencer::Type::FULL_SEQUENCE || _type == FrameSequencer::Type::FULL_SEQUENCE_REPEAT)
    {
        return next_frame();
    }
    if(_type == FrameSequencer::Type::INCREASING_STEP)
    {
        next_frame();
        return (int)std::distance(_precomp_sequence.begin(), _precomp_sequence_iterator);
    }

    return 0;
}
int FrameSequencer::current_frame ()
{
  return _frame;
}
