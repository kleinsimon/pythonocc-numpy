from typing import Type


class OCC_Wrapper_Base:
    _OCC_CLS: Type = None

    @classmethod
    def _get_cpp_pointer(cls, occ_obj) -> int:
        """Extracts the underlying C++ pointer from a SWIG wrapper object."""
        if not isinstance(occ_obj, cls._OCC_CLS):
            raise TypeError(f"Provided object is not a valid {cls._OCC_CLS.__name__} instance!")

        try:
            return int(occ_obj.this.this)
        except AttributeError:
            raise TypeError("Provided object is not a valid SWIG-wrapped OpenCASCADE instance!")