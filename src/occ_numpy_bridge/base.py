from typing import Type


class OCC_Wrapper_Base:
    _OCC_CLS: Type = None

    @classmethod
    def _get_cpp_pointer(cls, occ_obj, occ_type=None) -> int:
        """Extracts the underlying C++ pointer from a SWIG wrapper object."""
        if occ_type is None:
            occ_type = cls._OCC_CLS

        if not isinstance(occ_obj, occ_type):
            raise TypeError(f"Provided object is not a valid {occ_type.__name__} instance!")

        try:
            return int(occ_obj.this.this)
        except AttributeError:
            try:
                return int(occ_obj.this)
            except AttributeError:
                raise TypeError("Provided object is not a valid SWIG-wrapped OpenCASCADE instance!")