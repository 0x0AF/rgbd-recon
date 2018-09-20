#ifndef RGBD_RECON_FRAMESEQUENCER_H
#define RGBD_RECON_FRAMESEQUENCER_H

#include <vector>
#include "math.h"

class FrameSequencer
{
  public:
    enum class Type
    {
        FULL_SEQUENCE = 0,
        FULL_SEQUENCE_REPEAT = 1,
        INCREASING_STEP = 2
    };

    FrameSequencer(Type type, int frame_start, int frame_end);
    ~FrameSequencer() = default;

    int next_frame();
    int next_frame_position();
    bool is_finished();
    bool is_first_frame();
    int length();
    void rewind();

  private:
    std::vector<int> _precomp_sequence;
    std::vector<int>::iterator _precomp_sequence_iterator;
    bool _forward_motion;
    int _frame_start, _frame_end;
    int _frame;
    Type _type;
};

#endif // RGBD_RECON_FRAMESEQUENCER_H
