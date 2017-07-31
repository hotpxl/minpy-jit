# MinPy JIT

Bringing just-in-time compilation to MXNet.

## Development

Please adhere to [Google Python Style
Guide](https://google.github.io/styleguide/pyguide.html).

With [yapf](https://github.com/google/yapf) installed, run
`python-format.sh` before committing to format all code.

## Installation

Currently no installation is needed even there is a `setup.py` file
present. For examples, use `sys.path` manipulation to make it easier
to hack.

## Code review

If you have questions about other people's design decision or
implementation, it's best to submit a code review for discussion
purposes. They might also have reasons not so obvious. We write code
reviews directly in code comments so it can be easily tracked.

For example, if I want to code review Haoran's submission, I would add
`CR(haoran): questions I have` as comment to the location of
code. Haoran should respond by changing that `CR` to `XCR(yutian)`
(meaning cross-code review) and adding his response below. This way we
can keep track of replies and discussions. I will delete that `XCR`
when I accept the change or add another `XCR(haoran)` if I have more
to say.