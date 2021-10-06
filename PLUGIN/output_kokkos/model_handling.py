import madgraph.iolibs.export_cpp as export_cpp
import aloha.aloha_writers as aloha_writers
from madgraph import MG5DIR
import fractions
import os,logging
pjoin = os.path.join

logger = logging.getLogger('output_kokkos.model_handling')

class ALOHAWriterForGPU(aloha_writers.ALOHAWriterForGPU):
    
    extension = '.cc'
    prefix ='KOKKOS_FUNCTION'
    realoperator = '.real()'
    imagoperator = '.imag()'
    ci_definition = 'mgKokkos::complex cI = mgKokkos::complex(0., 1.);\n'
    
    type2def = {}    
    type2def['int'] = 'int '
    type2def['double'] = 'double '
    type2def['complex'] = 'mgKokkos::complex '
    type2def['pointer_vertex'] = '*' # using complex<double> * vertex)
    type2def['pointer_coup'] = ''

    # can over write some things using:
    def get_declaration_text(self,*args):
        return ''

class UFOModelConverterKokkos(export_cpp.UFOModelConverterCPP):
    
    #aloha_writer = 'cudac'
    cc_ext = 'cc'
        # Template files to use
    #include_dir = '.'
    #c_file_dir = '.'
    #param_template_h = 'cpp_model_parameters_h.inc'
    #param_template_cc = 'cpp_model_parameters_cc.inc'
    aloha_template_h = pjoin('kokkos','cpp_hel_amps_h.inc')
    aloha_template_cc = pjoin('kokkos','cpp_hel_amps_cc.inc')
    helas_h = pjoin('kokkos', 'helas.h')
    helas_cc = pjoin('kokkos', 'helas.cpp')

    def read_aloha_template_files(self, ext):
        """Read all ALOHA template files with extension ext, strip them of
        compiler options and namespace options, and return in a list"""

        path = pjoin(MG5DIR, 'aloha','template_files')
        out = []
        
        if ext == 'h':
            out.append(open(pjoin(path, self.helas_h)).read())
        else:
            out.append(open(pjoin(path, self.helas_cc)).read())
    
        return out

    def write_process_h_file(self, writer):
        
        replace_dict = super(UFOModelConverterKokkos,self).write_process_h_file(self, None)
        replace_dict['include_for_complex'] = '#include "mgKokkosTypes.h"'
        if writer:
            file = self.read_template_file(self.process_template_h) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict



import madgraph.iolibs.helas_call_writers as helas_call_writers
    
class GPUFOHelasCallWriter(helas_call_writers.GPUFOHelasCallWriter):

    def format_coupling(self, call):
        """Format the coupling so any minus signs are put in front"""
        return super().format_coupling(call)
        

    def get_external(self,wf, argument):
        """ formatting for ixxxx/ oxxxx /.... type of function (external ones) """
        return super().get_external(wf, argument)

    def generate_helas_call(self, argument):
        """Routine for automatic generation of C++ Helas calls
        according to just the spin structure of the interaction.

        First the call string is generated, using a dictionary to go
        from the spin state of the calling wavefunction and its
        mothers, or the mothers of the amplitude, to difenrentiate wich call is
        done.

        Then the call function is generated, as a lambda which fills
        the call string with the information of the calling
        wavefunction or amplitude. The call has different structure,
        depending on the spin of the wavefunction and the number of
        mothers (multiplicity of the vertex). The mother
        wavefunctions, when entering the call, must be sorted in the
        correct way - this is done by the sorted_mothers routine.

        Finally the call function is stored in the relevant
        dictionary, in order to be able to reuse the function the next
        time a wavefunction with the same Lorentz structure is needed.
        """
        return super().generate_helas_call(argument)


class OneProcessExporterKokkos(export_cpp.OneProcessExporterCPP):

    # Static variables (for inheritance)
    process_dir = '.'
    include_dir = '.'
    template_path = pjoin(os.path.dirname(export_cpp.__file__),'template_files')
    __template_path = pjoin(os.path.dirname(export_cpp.__file__),'template_files') 
    process_template_h = 'kokkos/process_h.inc'
    process_template_cc = 'kokkos/process_cc.inc'
    process_class_template = 'kokkos/process_class.inc'
    process_definition_template = 'kokkos/process_function_definitions.inc'
    process_wavefunction_template = 'cpp_process_wavefunctions.inc'
    process_sigmaKin_function_template = 'kokkos/process_sigmaKin_function.inc'
    single_process_template = 'kokkos/process_matrix.inc'
    cc_ext = 'cc'

    def __init__(self, *args, **opts):
        
        super(OneProcessExporterKokkos, self).__init__(*args, **opts)
        self.process_class = "CPPProcess"

    def generate_process_files(self):
        
        super(OneProcessExporterKokkos, self).generate_process_files()

        self.edit_check()
        self.edit_mgConfig()
        
        # add symbolic link for C++
        # files.ln(pjoin(self.path, 'gcheck_sa.cu'), self.path, 'check_sa.cc')
        # files.ln(pjoin(self.path, 'gCPPProcess.cu'), self.path, 'CPPProcess.cc')
        
    def edit_check(self):
        
        template = open(pjoin(self.template_path,'kokkos','check.cpp'),'r').read()
        replace_dict = {}
        replace_dict['nexternal'], _ = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['model'] = self.model_name
        replace_dict['numproc'] = len(self.matrix_elements)

        ff = open(pjoin(self.path, 'check.cpp'),'w')
        ff.write(template)
        ff.close()
        
    def edit_mgConfig(self):
        
        template = open(pjoin(self.template_path,'kokkos','mgKokkosConfig.h'),'r').read()
        replace_dict = {}
        nexternal, nincoming = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nincoming'] = nincoming
        replace_dict['noutcoming'] = nexternal - nincoming
        
        # Number of helicity combinations
        replace_dict['nbhel'] = \
                            self.matrix_elements[0].get_helicity_combinations()
        replace_dict['nwavefunc'] = \
                          self.matrix_elements[0].get_number_of_wavefunctions()
        replace_dict['wavefuncsize'] = 6
        
        ff = open(pjoin(self.path, '..','..','src','mgKokkosConfig.h'),'w')
        ff.write(template % replace_dict)
        ff.close()
        

    def get_initProc_lines(self, matrix_element, color_amplitudes):
        """Get initProc_lines for function definition for Pythia 8 .cc file"""

        initProc_lines = []

        initProc_lines.append("// Set external particle masses for this matrix element")

        for part in matrix_element.get_external_wavefunctions():
            initProc_lines.append("mME.push_back(pars->%s);" % part.get('mass'))
        #for i, colamp in enumerate(color_amplitudes):
        #    initProc_lines.append("jamp2[%d] = new double[%d];" % \
        #                          (i, len(colamp)))

        return "\n".join(initProc_lines)

    def get_reset_jamp_lines(self, color_amplitudes):
        """Get lines to reset jamps"""

        ret_lines = ""
        return ret_lines
    

    @staticmethod
    def coeff(ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
        """Returns a nicely formatted string for the coefficients in JAMP lines"""
    
        total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power
    
        if total_coeff == 1:
            if is_imaginary:
                return '+mgKokkos::complex(0,1)*'
            else:
                return '+'
        elif total_coeff == -1:
            if is_imaginary:
                return '-mgKokkos::complex(0,1)*'
            else:
                return '-'
    
        res_str = '%+i.' % total_coeff.numerator
    
        if total_coeff.denominator != 1:
            # Check if total_coeff is an integer
            res_str = res_str + '/%i.' % total_coeff.denominator
    
        if is_imaginary:
            res_str = res_str + '*mgKokkos::complex(0,1)'
    
        return res_str + '*'



    def get_process_function_definitions(self, write=True):
        """The complete Pythia 8 class definition for the process"""

        replace_dict = super(OneProcessExporterKokkos,self).get_process_function_definitions(write=False)


        replace_dict['ncouplings'] = len(self.couplings2order)
        replace_dict['ncouplingstimes2'] = 2 *  replace_dict['ncouplings']
        replace_dict['nparams'] = len(self.params2order)
        replace_dict['nmodels'] = replace_dict['nparams'] + replace_dict['ncouplings']
        replace_dict['coupling_list'] = ' '

        coupling = [''] * len(self.couplings2order)
        params = [''] * len(self.params2order)
        for coup, pos in self.couplings2order.items():
            coupling[pos] = coup
        coup_str = "static mgKokkos::complex tIPC[%s] = {pars->%s};\n"\
            %(len(self.couplings2order), ',pars->'.join(coupling))
        for para, pos in self.params2order.items():
            params[pos] = para            
        param_str = "static double tIPD[%s] = {pars->%s};\n"\
            %(len(self.params2order), ',pars->'.join(params))            
        
        
        replace_dict['assign_coupling'] = coup_str + param_str
        replace_dict['all_helicities'] = self.get_helicity_matrix(self.matrix_elements[0])
        replace_dict['all_helicities'] = replace_dict['all_helicities'] .replace("helicities", "tHel")
        
        file = self.read_template_file(self.process_definition_template) %\
               replace_dict

        return file

    def get_process_class_definitions(self, write=True):
        
        replace_dict = super(OneProcessExporterKokkos,self).get_process_class_definitions(write=False)

        replace_dict['nwavefuncs'] = replace_dict['wfct_size']
        replace_dict['namp'] = len(self.amplitudes.get_all_amplitudes())
        replace_dict['model'] = self.model_name
        
        replace_dict['sizew'] = self.matrix_elements[0].get_number_of_wavefunctions()
        replace_dict['nexternal'], _ = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['ncomb'] = len([x for x in self.matrix_elements[0].get_helicity_matrix()])
        
        replace_dict['all_sigma_kin_definitions'] = \
                          """// Calculate wavefunctions
                          KOKKOS_FUNCTION void calculate_wavefunctions(int ihel, double local_mom[%(nexternal)i][3],
                                        mgKokkos::complex amp[%(namp)d])
                          {
                          const int ncolor =  %(ncolor)d;
                          mgKokkos::complex jamp[ncolor];

                            mgKokkos::complex w[%(nwfct)d][%(sizew)d];
                            """ % \
                          {'nwfct':len(self.wavefunctions),
                          'sizew': replace_dict['wfct_size'],
                          'nexternal':replace_dict['nexternal'],
                          'namp':len(self.amplitudes),
                          'ncolor': len(self.matrix_elements[0].get_color_amplitudes())
                          }

        if write:
            file = self.read_template_file(self.process_class_template) % replace_dict
            return file
        else:
            return replace_dict
        
        
#     def get_calculate_wavefunctions(self, wavefunctions, amplitudes, write=True):
#         """Return the lines for optimized calculation of the
#         wavefunctions for all subprocesses"""
# 
#         raise Exception
#         replace_dict = {}
# 
#         replace_dict['nwavefuncs'] = len(wavefunctions)
#         
#         #ensure no recycling of wavefunction ! incompatible with some output
#         #for me in self.matrix_elements:
#         #    me.restore_original_wavefunctions()
# 
#         replace_dict['wavefunction_calls'] = "\n".join(\
#             self.helas_call_writer.get_wavefunction_calls(\
#             helas_objects.HelasWavefunctionList(wavefunctions)))
# 
#         replace_dict['amplitude_calls'] = "\n".join(\
#             self.helas_call_writer.get_amplitude_calls(amplitudes))
# 
#         if write:
#             file = self.read_template_file(self.process_wavefunction_template) % \
#                 replace_dict
#             return file
#         else:
#             return replace_dict
    
    def get_all_sigmaKin_lines(self, color_amplitudes, class_name):
        """Get sigmaKin_process for all subprocesses for Pythia 8 .cc file"""

        ret_lines = []
        if self.single_helicities:
            
            
            ret_lines.append(
                "KOKKOS_FUNCTION void calculate_wavefunctions(int ihel, const double* allmomenta,fptype &meHelSum )\n{"
                )

            ret_lines.append(" using namespace MG5_%s;" % self.model_name)
            ret_lines.append("mgDebug( 0, __FUNCTION__ );")
            ret_lines.append("mgKokkos::complex amp[1]; // was %i" % len(self.matrix_elements[0].get_all_amplitudes()))
            ret_lines.append("const int ncolor =  %i;" % len(color_amplitudes[0]))
            ret_lines.append("mgKokkos::complex jamp[ncolor];")
            ret_lines.append("// Calculate wavefunctions for all processes")
            ret_lines.append("using namespace MG5_%s;" % self.model_name)
            helas_calls = self.helas_call_writer.get_matrix_element_calls(\
                                                    self.matrix_elements[0],
                                                    color_amplitudes[0]
                                                    )
            logger.debug("only one Matrix-element supported?")
            self.couplings2order = self.helas_call_writer.couplings2order
            self.params2order = self.helas_call_writer.params2order
            nwavefuncs = self.matrix_elements[0].get_number_of_wavefunctions()
            ret_lines.append("mgKokkos::complex w[nwf][nw6];")


            ret_lines += helas_calls
            #ret_lines.append(self.get_calculate_wavefunctions(\
            #    self.wavefunctions, self.amplitudes))
            #ret_lines.append("}")
        else:
            ret_lines.extend([self.get_sigmaKin_single_process(i, me) \
                                  for i, me in enumerate(self.matrix_elements)])
        to_add = []
        to_add.extend([self.get_matrix_single_process(i, me,
                                                         color_amplitudes[i],
                                                         class_name) \
                                for i, me in enumerate(self.matrix_elements)])
        ret_lines.extend([self.get_matrix_single_process(i, me,
                                                         color_amplitudes[i],
                                                         class_name) \
                                for i, me in enumerate(self.matrix_elements)])
        return "\n".join(ret_lines)

    def write_process_h_file(self, writer):
        """Write the class definition (.h) file for the process"""
        
        replace_dict = super(OneProcessExporterKokkos, self).write_process_h_file(False)
        try:
            replace_dict['helamps_h'] = open(pjoin(self.path, os.pardir, os.pardir,'src','HelAmps_%s.h' % self.model_name)).read()
        except FileNotFoundError:
            replace_dict['helamps_h'] = "\n#include \"../../src/HelAmps_%s.h\"" % self.model_name
        
        if writer:
            file = self.read_template_file(self.process_template_h) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
    
    def write_process_cc_file(self, writer):
        """Write the class member definition (.cc) file for the process
        described by matrix_element"""
        
                
        replace_dict = super(OneProcessExporterKokkos, self).write_process_cc_file(False)
        try:
           replace_dict['hel_amps_def'] = open(pjoin(self.path, os.pardir, os.pardir,'src','HelAmps_%s.cc' % self.model_name)).read()
        except FileNotFoundError:
            replace_dict['hel_amps_def'] = "\n#include \"../../src/HelAmps_%s.cc\"" % self.model_name
            
        if writer:
            file = self.read_template_file(self.process_template_cc) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
