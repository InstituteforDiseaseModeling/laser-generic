import subprocess


def test_main():
    assert subprocess.check_output(["generic", "foo", "foobar", "bar"], text=True) == "foobar\n"
