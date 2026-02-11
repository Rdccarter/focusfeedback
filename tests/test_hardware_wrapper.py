from orca_focus.hardware import MclNanoZStage


class _MadlibStyleWrapper:
    def __init__(self) -> None:
        self.z = 1.25
        self.handle = None

    def init_handle(self) -> int:
        self.handle = 42
        return self.handle

    def single_read_n(self, axis: int, handle: int) -> float:
        assert axis == 3
        assert handle == 42
        return self.z

    def single_write_n(self, z_um: float, axis: int, handle: int) -> None:
        assert axis == 3
        assert handle == 42
        self.z = z_um

    def release_handle(self, handle: int) -> None:
        assert handle == 42


class _MadlibModule:
    MCL_Nanodrive = _MadlibStyleWrapper


class _SimpleWrapper:
    def __init__(self) -> None:
        self.z = 0.0

    def read_z(self, _axis: int) -> float:
        return self.z

    def write_z(self, z_um: float, _axis: int) -> None:
        self.z = z_um



def test_stage_madlib_style_wrapper_module() -> None:
    stage = MclNanoZStage(wrapper_module=_MadlibModule())

    assert stage.get_z_um() == 1.25
    stage.move_z_um(2.5)
    assert stage.get_z_um() == 2.5
    stage.close()


def test_stage_simple_wrapper_object() -> None:
    wrapper = _SimpleWrapper()
    stage = MclNanoZStage(wrapper_module=wrapper)

    stage.move_z_um(3.0)
    assert stage.get_z_um() == 3.0


def test_stage_context_manager() -> None:
    stage = MclNanoZStage()
    with stage as managed:
        managed.move_z_um(1.0)
    assert stage.get_z_um() == 1.0
