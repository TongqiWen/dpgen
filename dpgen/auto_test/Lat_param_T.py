import glob
import os
import json

import numpy as np
from monty.serialization import loadfn, dumpfn

from dpgen import dlog
from dpgen.auto_test.Property import Property


class Lat_param_T(Property):
    def __init__(self,
                 parameter):
        parameter['reproduce'] = parameter.get('reproduce', False)
        self.reprod = parameter['reproduce']
        self.T_start = parameter['T_start']
        self.T_end = parameter['T_end']
        self.T_step = parameter['T_step']

        default_cal_setting = {"input_prop": "lammps_input/lat_param_T/in.lammps"}
        if 'cal_setting' not in parameter:
            parameter['cal_setting'] = default_cal_setting
        self.cal_setting = parameter['cal_setting']
        parameter['cal_type'] = parameter.get('cal_type', 'relaxation')
        self.cal_type = parameter['cal_type']

        self.parameter = parameter

    def make_confs(self,
                   path_to_work,
                   path_to_equi,
                   do_refine=False):
        path_to_work = os.path.abspath(path_to_work)

        if os.path.exists(path_to_work):
            dlog.warning('%s already exists' % path_to_work)
        else:
            os.makedirs(path_to_work)
        path_to_equi = os.path.abspath(path_to_equi)

        struct_output_name = path_to_work.split('/')[-2]
        if 'start_confs_path' in self.parameter and os.path.exists(self.parameter['start_confs_path']):
            init_path_list = glob.glob(os.path.join(self.parameter['start_confs_path'], '*'))
            struct_init_name_list = []
            for ii in init_path_list:
                struct_init_name_list.append(ii.split('/')[-1])
            assert struct_output_name in struct_init_name_list
            path_to_equi = os.path.abspath(os.path.join(self.parameter['start_confs_path'],
                                                        struct_output_name, 'relaxation', 'relax_task'))

        equi_contcar = os.path.join(path_to_equi, 'CONTCAR')
        in_lammps_abs_path = os.path.abspath(self.cal_setting['input_prop'])
        file_abs_path = os.path.abspath(os.path.join(in_lammps_abs_path, '..'))
        lat_init_mod = os.path.join(file_abs_path, 'lat_param_init.mod')
        potential_mod = os.path.join(file_abs_path, 'potential.mod')
        lat_mod = os.path.join(file_abs_path, 'lat_param.mod')
        unitcell_mod = os.path.join(file_abs_path, 'unitcell.mod')
        sys_mod = os.path.join(file_abs_path, 'sys.mod')

        cwd = os.getcwd()
        os.chdir(path_to_work)
        if os.path.exists('POSCAR'):
            os.remove('POSCAR')
        os.symlink(os.path.relpath(equi_contcar), 'POSCAR')

        if os.path.exists('lat_param_init.mod'):
            os.remove('lat_param_init.mod')
        os.symlink(os.path.relpath(lat_init_mod), 'lat_param_init.mod')

        if os.path.exists('potential.mod'):
            os.remove('potential.mod')
        os.symlink(os.path.relpath(potential_mod), 'potential.mod')

        if os.path.exists('lat_param.mod'):
            os.remove('lat_param.mod')
        os.symlink(os.path.relpath(lat_mod), 'lat_param.mod')

        if os.path.exists('unitcell.mod'):
            os.remove('unitcell.mod')
        os.symlink(os.path.relpath(unitcell_mod), 'unitcell.mod')

        if os.path.exists('sys.mod'):
            os.remove('sys.mod')
        os.symlink(os.path.relpath(sys_mod), 'sys.mod')

        task_list = []

        print('gen Lat_param_T from ' + str(self.T_start) + ' to ' + str(self.T_end) + ' by every ' + str(self.T_step))
        task_num = 0

        while self.T_start + self.T_step * task_num < self.T_end:
            Temperature = self.T_start + task_num * self.T_step
            output_task = os.path.join(path_to_work, 'task.%06d' % task_num)
            os.makedirs(output_task, exist_ok=True)
            os.chdir(output_task)
            task_list.append(output_task)

            if os.path.exists('POSCAR'):
                os.remove('POSCAR')
            os.symlink('../POSCAR', 'POSCAR')

            if os.path.exists('lat_param_init.mod'):
                os.remove('lat_param_init.mod')
            os.symlink('../lat_param_init.mod', 'lat_param_init.mod')

            if os.path.exists('potential.mod'):
                os.remove('potential.mod')
            os.symlink('../potential.mod', 'potential.mod')

            if os.path.exists('lat_param.mod'):
                os.remove('lat_param.mod')
            os.symlink('../lat_param.mod', 'lat_param.mod')

            if os.path.exists('unitcell.mod'):
                os.remove('unitcell.mod')
            os.symlink('../unitcell.mod', 'unitcell.mod')

            if os.path.exists('sys.mod'):
                os.remove('sys.mod')
            os.symlink('../sys.mod', 'sys.mod')

            with open('in.lammps', 'w+') as fout:
                with open(in_lammps_abs_path, 'r') as fin:
                    for ii in range(15):
                        print(fin.readline().strip('\n'), file=fout)
                    fin.readline()
                    print("variable run_temp string", str(Temperature), file=fout)
                    for line in fin:
                        print(line.strip('\n'), file=fout)

            with open('init.mod', 'w+') as fout:
                with open(os.path.join(file_abs_path, 'init.mod'), 'r') as fin:
                    for ii in range(20):
                        print(fin.readline().strip('\n'), file=fout)
                    fin.readline()
                    print("variable temp equal", str(Temperature), '# temperature of initial sample', file=fout)
                    for line in fin:
                        print(line.strip('\n'), file=fout)

            task_num += 1
            os.chdir(cwd)
        return task_list

    def post_process(self, task_list):
        pass

    def task_type(self):
        return self.parameter['type']

    def task_param(self):
        return self.parameter

    def _compute_lower(self,
                       output_file,
                       all_tasks,
                       all_res):
        output_file = os.path.abspath(output_file)
        res_data = {}
        ptr_data = "conf_dir: " + os.path.dirname(output_file) + "\n"

        if not self.reprod:
            ptr_data += ' Temperature(K)  a(A)  b(A)  c(A)  c/a\n'
            for ii in range(len(all_tasks)):
                # vol = self.vol_start + ii * self.vol_step
                with open(os.path.join(all_tasks[ii], 'lat_param_finite_temp.mod'), 'r') as fin:
                    task_result = fin.readlines()
                for jj in task_result:
                    for kk in jj.split():
                        if 'lat' in kk and '_a_' in kk:
                            lat_a = float(jj.split()[3])
                        elif 'lat' in kk and '_b_' in kk:
                            lat_b = float(jj.split()[3])
                        elif 'lat' in kk and '_c_' in kk:
                            lat_c = float(jj.split()[3])
                        elif 'lat' in kk and '_k_' in kk:
                            lat_k = float(jj.split()[3])
                temperature = self.T_start + ii * self.T_step
                ptr_data += '%4d  %8.4f  %8.4f  %8.4f  %8.4f\n' % (temperature, lat_a, lat_b, lat_c, lat_k)
                res_data[temperature] = [lat_a, lat_b, lat_c, lat_k]
                # ptr_data += '%7.3f  %8.4f \n' % (vol, all_res[ii]['energy'] / len(all_res[ii]['force']))
            with open(output_file, 'w') as fp:
                json.dump(res_data, fp, indent=4)

        return res_data, ptr_data
