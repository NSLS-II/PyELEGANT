import os
import pickle
import subprocess
import errno

from pylatex import *
import pylatex as plx

class CombinedMathNormalText(plx.Math):
    """"""
    def __init__(self):

        super().__init__(inline=True, data=[])

    def _conv_str_to_math(self, s):
        """"""

        math_str = ''.join([f'{{{v}}}' if v != ' ' else r'\ ' for v in s])
        math_str = math_str.replace('_', r'\_')
        math_str = math_str.replace('&', r'\&')
        math_str = math_str.replace('}{', '')
        math_str = math_str.replace('{', r'\mathrm{')

        return plx.NoEscape(math_str)

    def copy(self):
        """"""

        return pickle.loads(pickle.dumps(self))

    def __add__(self, other):
        """"""

        copy = CombinedMathNormalText()
        copy.data.extend(self.data)

        copy.append(other)

        return copy

    def __radd__(self, left):
        """"""

        if isinstance(left, str):
            copy = CombinedMathNormalText()
            copy.data.append(self._conv_str_to_math(left))
            left = copy

        return left.__add__(self)

    def append(self, obj):
        """"""

        if isinstance(obj, str):
            self.data.append(self._conv_str_to_math(obj))
        else:
            assert isinstance(obj, plx.Math)
            self.data.extend([plx.NoEscape(s) for s in obj.data])

    def extend(self, obj_list):
        """"""

        for i, obj in enumerate(obj_list):
            if isinstance(obj, str):
                self.data.append(self._conv_str_to_math(obj))
            else:
                assert isinstance(obj, plx.Math)
                self.data.extend([plx.NoEscape(s) for s in obj.data])

    def clear(self):
        """"""

        self.data.clear()

    def dumps_NoEscape(self):
        """"""

        return plx.NoEscape(self.dumps())

    def dumps_for_caption(self):
        """"""

        return self.dumps_NoEscape()

class MathText(CombinedMathNormalText):
    """"""

    def __init__(self, r_str):

        super().__init__()
        self.data.append(plx.NoEscape(r_str))

        self.r_str = r_str # without this, print() of this object will fail

class FigureForMultiPagePDF(plx.Figure):
    """"""

    _latex_name = "figure"

    def add_image(self, filename, *, width=plx.NoEscape(r'0.8\textwidth'),
                  page=None, placement=plx.NoEscape(r'\centering')):
        """Add an image to the figure.

        Args
        ----
        filename: str
            Filename of the image.
        width: str
            The width of the image
        page: int
            The page number of the PDF file for the image
        placement: str
            Placement of the figure, `None` is also accepted.

        """

        if width is not None:
            if self.escape:
                width = plx.escape_latex(width)

            image_options = 'width=' + str(width)

        if page is not None:
            image_options = [image_options, f'page={page:d}']

        if placement is not None:
            self.append(placement)

        self.append(plx.StandAloneGraphic(
            image_options=image_options,
            filename=plx.utils.fix_filename(filename)))

class SubFigureForMultiPagePDF(FigureForMultiPagePDF):
    """A class that represents a subfigure from the subcaption package."""

    _latex_name = "subfigure"

    packages = [plx.Package('subcaption')]

    #: By default a subfigure is not on its own paragraph since that looks
    #: weird inside another figure.
    separate_paragraph = False

    _repr_attributes_mapping = {
        'width': 'arguments',
    }

    def __init__(self, width=plx.NoEscape(r'0.45\linewidth'), **kwargs):
        """
        Args
        ----
        width: str
            Width of the subfigure itself. It needs a width because it is
            inside another figure.

        """

        super().__init__(arguments=width, **kwargs)

    def add_image(self, filename, *, width=plx.NoEscape(r'\linewidth'), page=None,
                  placement=None):
        """Add an image to the subfigure.

        Args
        ----
        filename: str
            Filename of the image.
        width: str
            Width of the image in LaTeX terms.
        page: int
            The page number of the PDF file for the image
        placement: str
            Placement of the figure, `None` is also accepted.
        """

        super().add_image(filename, width=width, page=page, placement=placement)

def generate_pdf_w_reruns(
    doc, filepath=None, *, clean=True, clean_tex=True,
    compiler=None, compiler_args=None, silent=True, nMaxReRuns=10):
    """
    When the output of running LaTeX contains "Rerun LaTeX.", this function
    will re-run latex until this message disappears up to "nMaxReRuns" times.


    Generate a pdf file from the document.

    Args
    ----
    filepath: str
        The name of the file (without .pdf), if it is `None` the
        ``default_filepath`` attribute will be used.
    clean: bool
        Whether non-pdf files created that are created during compilation
        should be removed.
    clean_tex: bool
        Also remove the generated tex file.
    compiler: `str` or `None`
        The name of the LaTeX compiler to use. If it is None, PyLaTeX will
        choose a fitting one on its own. Starting with ``latexmk`` and then
        ``pdflatex``.
    compiler_args: `list` or `None`
        Extra arguments that should be passed to the LaTeX compiler. If
        this is None it defaults to an empty list.
    silent: bool
        Whether to hide compiler output
    """

    rm_temp_dir = plx.utils.rm_temp_dir
    CompilerError = plx.errors.CompilerError

    if compiler_args is None:
        compiler_args = []

    filepath = doc._select_filepath(filepath)
    filepath = os.path.join('.', filepath)

    cur_dir = os.getcwd()
    dest_dir = os.path.dirname(filepath)
    basename = os.path.basename(filepath)

    if basename == '':
        basename = 'default_basename'

    os.chdir(dest_dir)

    doc.generate_tex(basename)

    if compiler is not None:
        compilers = ((compiler, []),)
    else:
        latexmk_args = ['--pdf']

        compilers = (
            ('latexmk', latexmk_args),
            ('pdflatex', [])
        )

    main_arguments = ['--interaction=nonstopmode', basename + '.tex']

    os_error = None

    for compiler, arguments in compilers:
        command = [compiler] + arguments + compiler_args + main_arguments

        if compiler == 'latexmk':
            actual_nMaxReRuns = 1
        else:
            actual_nMaxReRuns = nMaxReRuns

        check_next_compiler = False

        for iLaTeXRun in range(actual_nMaxReRuns):

            try:
                output = subprocess.check_output(command,
                                                 stderr=subprocess.STDOUT)
            except (OSError, IOError) as e:
                # Use FileNotFoundError when python 2 is dropped
                os_error = e

                if os_error.errno == errno.ENOENT:
                    # If compiler does not exist, try next in the list
                    check_next_compiler = True
                    break
                    #continue
                raise
            except subprocess.CalledProcessError as e:
                # For all other errors print the output and raise the error
                print(e.output.decode())
                raise
            else:
                if not silent:
                    print(output.decode())

            if (compiler == 'pdflatex') and ('Rerun LaTeX.' in output.decode()):
                print(f'\n\n*** LaTeX rerun instruction detected. '
                      f'Re-running LaTeX (Attempt #{iLaTeXRun+2:d})\n\n')
                continue
            else:
                break

        if check_next_compiler:
            continue

        if clean:
            try:
                # Try latexmk cleaning first
                subprocess.check_output(['latexmk', '-c', basename],
                                        stderr=subprocess.STDOUT)
            except (OSError, IOError, subprocess.CalledProcessError) as e:
                # Otherwise just remove some file extensions.
                extensions = ['aux', 'log', 'out', 'fls',
                              'fdb_latexmk']

                for ext in extensions:
                    try:
                        os.remove(basename + '.' + ext)
                    except (OSError, IOError) as e:
                        # Use FileNotFoundError when python 2 is dropped
                        if e.errno != errno.ENOENT:
                            raise
            rm_temp_dir()

        if clean_tex:
            os.remove(basename + '.tex')  # Remove generated tex file

        # Compilation has finished, so no further compilers have to be
        # tried
        break

    else:
        # Notify user that none of the compilers worked.
        raise(CompilerError(
            'No LaTex compiler was found\n' +
            'Either specify a LaTex compiler ' +
            'or make sure you have latexmk or pdfLaTex installed.'
        ))

    os.chdir(cur_dir)

def NewParagraph():
    """"""

    return plx.Command('par')

def ClearPage():
    """"""

    return plx.Command('clearpage')
