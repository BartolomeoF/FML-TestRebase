#ifndef MPIPARTICLES_HEADER
#define MPIPARTICLES_HEADER

#include <vector>
#include <ios>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdio>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <FML/Global/Global.h>

namespace FML {
  namespace PARTICLE {

    //===========================================================
    /// 
    /// A container class for holding particles that are distributed
    /// across many CPUs and to easily deal with communication
    /// of these particles if they cross the domain boundary (simply
    /// class communicate_particles() at any time to do this)
    ///
    /// Contains methods for setting up MPIParticles from a set of particles
    /// or creating regular grids of particles and so on
    ///
    /// Templated on particle class. Particle must at a minimum have the methods:
    ///
    ///    auto *get_pos()                         : Ptr to position
    ///
    ///    int get_ndim()                          : How many dimensions pos have
    ///
    ///    get_particle_byte_size()                : How many bytes does the particle store
    ///
    ///    append_to_buffer(char *)   : append all the particle data to a char array moving buffer forward as we read
    ///
    ///    assign_from_buffer(char *) : assign data to a particle after it has been recieved moving buffer forward as we do this
    ///
    /// Compile time defines:
    ///
    ///    USE_MPI  : Use MPI
    ///
    ///    USE_OMP  : Use OpenMP
    ///
    ///    DEBUG_MPIPARTICLES    : Show some info when running
    ///
    /// External variables/methods we rely on:
    ///
    ///    int ThisTask;
    ///
    ///    int NTasks;
    ///
    ///    assert_mpi(Expr, Msg)
    ///
    ///    T power(T base, int exponent);
    ///
    //===========================================================

    template<class T>
      class MPIParticles {
        private:

          // Particle container
          //std::vector<T> p;
          Vector<T> p;

          // Info about the particle distribution
          size_t NpartTotal;          // Total number of particles across all tasks
          size_t NpartLocal_in_use;   // Number of particles in use (total allocated given by p.size())

          // The range of x-values in [0,1] that belongs to each task
          std::vector<double> x_min_per_task; 
          std::vector<double> x_max_per_task;

          // If we create particles from scratch according to a grid
          double buffer_factor;       // Allocate a factor of this more particles than needed

          // If we created a uniform grid of particles
          int Npart_1D;               // Total number of particles slices
          int Local_Npart_1D;         // Number of slices in current task
          int Local_p_start;          // Start index of local particle slice in [0,Npart_1D)

        public:

          // Iterator for loopping through all the active particles
          // i.e. allow for(auto &&p: mpiparticles)
          class iterator {
            public:
              iterator(T * ptr): ptr(ptr){}
              iterator operator++() { ++ptr; return *this; }
              bool operator!=(const iterator & other) { return ptr != other.ptr; }
              T& operator*() { return *ptr; }
            private:
              T* ptr;
          };
          iterator begin() { return iterator(p.data()); }
          iterator end() { return iterator(p.data() + NpartLocal_in_use); }

          // Create MPIParticles from a set of particles
          // NumParts in the number of particles in part
          // xmin and xmax corresponds to the x-range in [0,1] the current task is responsible for
          // Set all_tasks_has_the_same_particles if all tasks has the same set of particles. 
          // If only one or more task has particles then set all_tasks_has_the_same_particles = false.
          // NB: if all_tasks_has_the_same_particles is false then we assume all tasks that has particles has distinct particles
          // We only keep the particles in part that are in the correct x-range if all_tasks_has_the_same_particles is true
          // nallocate is the total amount of particles we allocate for locally (we do buffered read if only one task
          // reads particles to avoid having to allocate too much). 
          void create(T *part, size_t NumParts, size_t nallocate, double xmin_local, double xmax_local, bool all_tasks_has_the_same_particles);

          // Create Npart_1D^3 particles in a rectangular grid spread across all tasks
          // buffer_factor is how much extra to allocate in case particles moves
          // xmin, xmax specifies the domain in [0,1] that the current task is responsible for
          // This method only sets the positions of the particles not id or vel or anything else
          void create_particle_grid(int Npart_1D, double buffer_factor, double xmin_local, double xmax_local);

          MPIParticles();

          // Get reference to particle vector. NB: due to we allow a buffer the size of the vector returned is 
          // not equal to the 
          Vector<T> & get_particles();
          T * get_particles_ptr();

          // Access particles through indexing operator
          T& operator [](size_t i);
          T & get_part(int i);

          // Swap particles
          void swap_particles(T &a, T&b);

          // Get position and velocity for particle ipart
          double get_pos(int ipart, int i);
          double get_vel(int ipart, int i); 

          // Some useful info
          size_t get_npart_total() const;
          size_t get_npart() const;
          size_t get_particle_byte_size();

          // If we created a uniform distribution of particles
          int get_local_np() const;
          int get_local_p_start() const;
          int get_npart_1d() const;

          // Communicate particles across CPU boundaries
          void copy_over_recieved_data(std::vector<char> &recv_buffer, int Npart_recieved);
          void communicate_particles();

          // xmin and xmax for each task
          std::vector<double> get_x_min_per_task();
          std::vector<double> get_x_max_per_task();

          void free();

          // For memory logging
          void add_memory_label(std::string name);

          // Write / read from file
          void dump_to_file(std::string fileprefix, size_t max_bytesize_buffer = 100 * 1000 * 1000);
          void load_from_file(std::string fileprefix, size_t max_bytesize_buffer = 100 * 1000 * 1000);

          // Show some info
          void info();
      };

    template<class T>
      void MPIParticles<T>::info(){
        if(FML::ThisTask > 0) return;
        T tmp;
        auto bytes = tmp.get_particle_byte_size();
        auto NDIM = tmp.get_ndim();
        double memory_in_mb = p.size() * bytes/1e6; 
        std::cout << "\n========================================================\n";
        std::cout << "MPIParticles Particles have Ndim: [" << NDIM << "]" << "\n";
        std::cout << "We have allocated " << memory_in_mb << " MB of memory per task\n"; 
        std::cout << "NParticles local task  " << NpartLocal_in_use      << "\n";
        std::cout << "NParticles all tasks   " << NpartTotal             << "\n";
        std::cout << "The buffer is " << NpartLocal_in_use/double(p.size()) * 100 << "% filled\n";
        std::cout << "========================================================\n\n";
      }

    template<class T>
      void MPIParticles<T>::add_memory_label([[maybe_unused]] std::string name){
#ifdef MEMORY_LOGGING
        FML::MemoryLog::get()->add_label(p.data(), p.capacity(), name);
#endif
      }

    template<class T>
      void MPIParticles<T>::free(){
        p.clear();
        p.shrink_to_fit();
      }

    template<class T>
      void MPIParticles<T>::create(
          T *part,
          size_t NumPartinpart,
          size_t nallocate, 
          double xmin, 
          double xmax, 
          bool all_tasks_has_the_same_particles){

        if(NTasks == 1) all_tasks_has_the_same_particles = true;

        // Set the xmin/xmax
        x_min_per_task = std::vector<double>(NTasks,0.0);
        x_max_per_task = std::vector<double>(NTasks,0.0);
        x_min_per_task[ThisTask] = xmin;
        x_max_per_task[ThisTask] = xmax;
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, x_min_per_task.data(), NTasks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, x_max_per_task.data(), NTasks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

#ifdef DEBUG_MPIPARTICLES
        if(ThisTask == 0){
          for(int i = 0; i < NTasks; i++){
            std::cout << "Task: " << i << " / " << NTasks << " xmin: " << x_min_per_task[i] << " xmax: " << x_max_per_task[i] << "\n";
          }
        }
#endif

        // Allocate memory
        p.resize(nallocate);
        add_memory_label("MPIPartices::create");

        // If all tasks has the same particles
        // we read all the particles and only keep the particles in range
        if(all_tasks_has_the_same_particles){
          size_t count = 0;
          for(size_t i = 0; i < NumPartinpart; i++){
            auto *pos = part[i].get_pos();
            if(pos[0] >= xmin and pos[0] < xmax){
              p[count] = part[i];
              count++;
            }
          }

          // Check that we are not past allocation limit
          if(count > nallocate) {
            assert_mpi(false, 
                "[MPIParticle::create] Reached allocation limit. Increase nallocate\n");
          }

          NpartLocal_in_use = count;
        }

#ifdef USE_MPI
        std::cout << std::flush;
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // If only one or more task has particles then read in batches and communicate as we go along
        // just in case the total amount of particles are too large
        if(!all_tasks_has_the_same_particles){

          const int nmax_per_batch = nallocate;
          // Read in batches
          size_t start = 0;
          size_t count = 0;
          bool more_to_process_globally = true;
          bool more_to_process_locally = NumPartinpart > 0;
          while(more_to_process_globally){
            if(more_to_process_locally){
              size_t nbatch = start + nmax_per_batch < NumPartinpart ? start + nmax_per_batch : NumPartinpart;
              more_to_process_locally = nbatch < NumPartinpart;

              for(size_t i = start; i < nbatch; i++){
                p[count] = part[i];
                count++;
                if(count > nallocate){
                  assert_mpi(false, 
                      "[MPIParticle::create] Reached allocation limit. Increase nallocate\n");
                }
              }
              start = nbatch;
            }

            // Set the number of particles read
            NpartLocal_in_use = count;

#ifdef DEBUG_MPIPARTICLES
            std::cout << "Task: " << ThisTask << " NpartLocal_in_use: " <<  NpartLocal_in_use << " precomm\n";
#endif

            // Send particles to where they belong
            communicate_particles();

#ifdef DEBUG_MPIPARTICLES
            std::cout << "Task: " << ThisTask << " NpartLocal_in_use: " <<  NpartLocal_in_use << " postcomm\n";
#endif

            // Update how many particles we now have
            count = NpartLocal_in_use;

#ifdef USE_MPI
            // The while loop continues until all tasks are done reading particles
            int moretodo = more_to_process_locally ? 1 : 0;
            MPI_Allreduce(MPI_IN_PLACE, &moretodo, NTasks, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            more_to_process_globally = (moretodo == 1);
#endif
          }
        }

        // Set total number of particles
        NpartTotal = NpartLocal_in_use;
#ifdef USE_MPI
        long long int np = NpartLocal_in_use;
        MPI_Allreduce(MPI_IN_PLACE, &np, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        NpartTotal = np;
#endif

#ifdef DEBUG_MPIPARTICLES
        std::cout << "Task: " << ThisTask << " NpartLocal_in_use: " <<  NpartLocal_in_use << "\n";
#endif
      }

    template<class T>
      T & MPIParticles<T>::get_part(int i) { 
        return p[i]; 
      }

    template<class T>
      size_t MPIParticles<T>::get_npart() const { 
        return NpartLocal_in_use; 
      }

    template<class T>
      size_t MPIParticles<T>::get_npart_total() const{
        return NpartTotal;
      }

    template<class T>
      int MPIParticles<T>::get_npart_1d() const{ 
        return Npart_1D; 
      }

    template<class T>
      int MPIParticles<T>::get_local_np() const { 
        return Local_Npart_1D; 
      }

    template<class T>
      int MPIParticles<T>::get_local_p_start() const {
        return Local_p_start; 
      }

    template<class T>
      Vector<T> & MPIParticles<T>::get_particles(){
        return p;
      }

    template<class T>
      T * MPIParticles<T>::get_particles_ptr(){ 
        return p.data(); 
      }

    template<class T>
      std::vector<double> MPIParticles<T>::get_x_min_per_task(){
        return x_min_per_task;
      }

    template<class T>
      std::vector<double> MPIParticles<T>::get_x_max_per_task(){
        return x_max_per_task;
      }

    template<class T>
      T& MPIParticles<T>::operator [](size_t i) {
        return p[i];
      }

    template<class T>
      void MPIParticles<T>::swap_particles(T &a, T&b){
        T tmp = a;
        a = b;
        b = tmp;
      }

    template<class T>
      MPIParticles<T>::MPIParticles() : 
        p(0),
        NpartTotal(0), 
        NpartLocal_in_use(0), 
        buffer_factor(1.0), 
        Npart_1D(0), 
        Local_Npart_1D(0), 
        Local_p_start(0)
    {
    }

    template<class T>
      void MPIParticles<T>::create_particle_grid(int Npart_1D, double buffer_factor, double xmin_local, double xmax_local){
        this->Npart_1D = Npart_1D;
        this->buffer_factor = buffer_factor;

        // Use the local xmin,xmax values to compute how many slices per task
        int imin = 0;
        while(imin / double(Npart_1D) < xmin_local){
          imin++;
        }
        int imax = imin;
        while(imax / double(Npart_1D) < xmax_local){
          imax++;
        }
        Local_p_start = imin;
        Local_Npart_1D = imax - imin;

        // Sanity check
        std::vector<int> Local_Npart_1D_per_task(NTasks,0);
        Local_Npart_1D_per_task[ThisTask] = Local_Npart_1D;
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, Local_Npart_1D_per_task.data(), NTasks, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
        int Local_p_start_computed = 0, Npart_1D_computed = 0;
        for(int i = 0; i < NTasks; i++){
          if(i < ThisTask) Local_p_start_computed += Local_Npart_1D_per_task[i];
          Npart_1D_computed += Local_Npart_1D_per_task[i];
        }
        assert_mpi(Npart_1D_computed == Npart_1D, 
            "[MPIParticles::create_particle_grid] Computed Npart does not match Npart\n");
        assert_mpi(Local_p_start_computed == Local_p_start, 
            "[MPIParticles::create_particle_grid] Computed Local_p_start does not match Local_p_start\n");

        // Get the min and max x-positions (in [0,1]) that each of the tasks is responsible for
        x_min_per_task = std::vector<double>(NTasks,0.0);
        x_max_per_task = std::vector<double>(NTasks,0.0);

        // If we don't have xmin,xmax availiable 
        //x_min_per_task[ThisTask] = Local_p_start / double(Npart_1D);
        //x_max_per_task[ThisTask] = (Local_p_start + Local_Npart_1D) / double(Npart_1D);

        // Fetch these values
        x_min_per_task[ThisTask] = xmin_local;
        x_max_per_task[ThisTask] = xmax_local;
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, x_min_per_task.data(), NTasks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, x_max_per_task.data(), NTasks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

#ifdef DEBUG_MPIPARTICLES
        // Output some info
        for(int taskid = 0; taskid < NTasks; taskid++){
          if(ThisTask == 0){
            std::cout << "Task[" << taskid << "]  xmin: " << x_min_per_task[taskid] <<" xmax: " << x_max_per_task[taskid] << "\n";
          }
        }
#endif

        // Total number of particles
        int Ndim = p.data()->get_ndim();
        NpartTotal = power(size_t(Npart_1D),Ndim);
        NpartLocal_in_use = Local_Npart_1D * power(size_t(Npart_1D),Ndim-1);

        // Allocate particle struct
        size_t NpartToAllocate = size_t(NpartLocal_in_use * buffer_factor);
        p.resize(NpartToAllocate);
        add_memory_label("MPIPartices::create_particle_grid");

        // Initialize the coordinate to the first cell in the local grid
        const int ndim = p[0].get_ndim();
        std::vector<double> Pos(ndim,0.0);
        std::vector<int> coord(ndim,0);
        coord[0] = Local_p_start;
        for(size_t i = 0; i < NpartLocal_in_use; i++){
          auto *Pos = p[i].get_pos();

          // Position regular grid
          for(int idim = 0; idim < ndim; idim++){
            Pos[idim] = coord[idim] / double(Npart_1D);
          }

          // This is adding 1 very time in base Npart_1D storing the digits in reverse order in [coord]
          int idim = ndim-1;
          while(++coord[idim] == Npart_1D){
            coord[idim--] = 0;
            if(idim < 0) break;
          }
        }
      }

    template<class T>
      double MPIParticles<T>::get_pos(int ipart, int i) {
        return  p[ipart].get_pos()[i];
      }

    template<class T>
      double MPIParticles<T>::get_vel(int ipart, int i) {
        return p[ipart].get_vel()[i];
      }

    template<class T>
      size_t MPIParticles<T>::get_particle_byte_size(){
        T a;
        return a.get_particle_byte_size();
      }

    template<class T>
      void MPIParticles<T>::copy_over_recieved_data(std::vector<char> &recv_buffer, int Npart_recv){
        assert_mpi(NpartLocal_in_use + Npart_recv <= p.size(), 
            "[MPIParticles::copy_over_recieved_data] Too many particles recieved! Increase buffer\n");

        size_t bytes_per_particle = get_particle_byte_size();
        for(int i = 0; i < Npart_recv; i++){
          p[NpartLocal_in_use+i].assign_from_buffer(&recv_buffer[bytes_per_particle * i]);
        }

        // Update the total number of particles in use
        NpartLocal_in_use += Npart_recv;
      }

    template<class T>
      void MPIParticles<T>::communicate_particles(){
#ifdef USE_MPI

        // The number of particles we start with
        size_t NpartLocal_in_use_pre_comm = NpartLocal_in_use;

#ifdef DEBUG_MPIPARTICLES
        if(ThisTask == 0){
          std::cout << "Communicating particles task: " << ThisTask << " Nparticles: " << NpartLocal_in_use_pre_comm << "\n" << std::flush;
        }
#endif

        // Count how many particles to send to each task
        // and move the particles to be send to the back of the array
        // and reduce the NumPartLocal_in_use if a partice is to be sent
        // After this is done we have all the particles to be send in 
        // location [NpartLocal_in_use, NpartLocal_in_use_pre_comm)
        std::vector<int> n_to_send(NTasks,0);
        std::vector<int> n_to_recv(NTasks,0);
        size_t i = 0;
        while(i < NpartLocal_in_use){
          double x = p[i].get_pos()[0];
          if(x >= x_max_per_task[ThisTask]){
            int taskid = ThisTask;
            while(1){
              ++taskid;
              if(x < x_max_per_task[taskid]) break;
            }

            n_to_send[taskid]++;
            swap_particles(p[i],p[--NpartLocal_in_use]);

          } else if(x < x_min_per_task[ThisTask]) {
            int taskid = ThisTask;
            while(1){
              --taskid;
              if(x >= x_min_per_task[taskid]) break;
            } 

            n_to_send[taskid]++;
            swap_particles(p[i],p[--NpartLocal_in_use]);

          } else {
            i++;
          }
        }

        // Communicate to get how many to recieve from each task
        for(int i = 1; i < NTasks; i++){
          int send_request_to  = (ThisTask + i)         % NTasks;
          int get_request_from = (ThisTask - i + NTasks)% NTasks;

          // Send to the right, recieve from left
          MPI_Status status;
          MPI_Sendrecv(
              &n_to_send[send_request_to],  1, MPI_INT, send_request_to,  0,
              &n_to_recv[get_request_from], 1, MPI_INT, get_request_from, 0, 
              MPI_COMM_WORLD, &status);

        }

#ifdef DEBUG_MPIPARTICLES
        // Show some info
        if(ThisTask == 0){
          for(int i = 0; i < NTasks; i++){
            std::cout << "Task " << ThisTask << " send to   " << i << " : " << n_to_send[i] << "\n" << std::flush;
            std::cout << "Task " << ThisTask << " recv from " << i << " : " << n_to_recv[i] << "\n" << std::flush;
          }
        }
#endif

        // Total number to send and recv
        size_t ntot_to_send = 0;
        size_t ntot_to_recv = 0;
        for(int i = 0; i < NTasks; i++){
          ntot_to_send += n_to_send[i];
          ntot_to_recv += n_to_recv[i];
        }

        // Sanity check
        assert_mpi(NpartLocal_in_use_pre_comm == NpartLocal_in_use + ntot_to_send, 
            "[MPIParticles::communicate_particles] Number to particles to communicate does not match\n");

        // Allocate send buffer
        size_t byte_per_particle = get_particle_byte_size();
        std::vector<char> send_buffer(byte_per_particle * ntot_to_send);
        std::vector<char> recv_buffer(byte_per_particle * ntot_to_recv);

        // Pointers to each send-recv place in the send-recv buffer
        std::vector<size_t> offset_in_send_buffer(NTasks,0);
        std::vector<size_t> offset_in_recv_buffer(NTasks,0);
        std::vector<char*> send_buffer_by_task(NTasks,send_buffer.data());
        std::vector<char*> recv_buffer_by_task(NTasks,recv_buffer.data());
        for(int i = 1; i < NTasks; i++){
          offset_in_send_buffer[i] = offset_in_send_buffer[i-1] + n_to_send[i-1] * byte_per_particle;
          offset_in_recv_buffer[i] = offset_in_recv_buffer[i-1] + n_to_recv[i-1] * byte_per_particle;
          send_buffer_by_task[i]   = &send_buffer.data()[offset_in_send_buffer[i]];
          recv_buffer_by_task[i]   = &recv_buffer.data()[offset_in_recv_buffer[i]];
        }

        // Gather particle data
        for(size_t i = 0; i < ntot_to_send; i++){
          size_t index = NpartLocal_in_use+i;
          double x = p[index].get_pos()[0];
          if(x >= x_max_per_task[ThisTask]){
            int taskid = ThisTask;
            while(1){
              ++taskid;
              if(x < x_max_per_task[taskid]) break;
            }

            p[index].append_to_buffer(send_buffer_by_task[taskid]);
            send_buffer_by_task[taskid] += byte_per_particle;

          } else if(x < x_min_per_task[ThisTask]) {
            int taskid = ThisTask;
            while(1){
              --taskid;
              if(x >= x_min_per_task[taskid]) break;
            }

            p[index].append_to_buffer(send_buffer_by_task[taskid]);
            send_buffer_by_task[taskid] += byte_per_particle;

          } else {

            // We should not be here as particles are moved
            assert_mpi(false, 
                "[MPIParticles::communicate_particles] Error in communicate_particles. After moving particles we still have particles out of range\n");
          }
        }

        // We changed the send-recv pointers above so reset them
        for(int i = 0; i < NTasks; i++){
          send_buffer_by_task[i]   = &send_buffer.data()[offset_in_send_buffer[i]];
          recv_buffer_by_task[i]   = &recv_buffer.data()[offset_in_recv_buffer[i]];
        }

        // Communicate the particle data
        for(int i = 1; i < NTasks; i++){
          int send_request_to  = (ThisTask + i)         % NTasks;
          int get_request_from = (ThisTask - i + NTasks)% NTasks;

          // Send to the right, recieve from left
          MPI_Status status;
          MPI_Sendrecv(
              send_buffer_by_task[send_request_to],  
              n_to_send[send_request_to] * byte_per_particle,  
              MPI_CHAR, send_request_to,  0,
              recv_buffer_by_task[get_request_from], 
              n_to_recv[get_request_from] * byte_per_particle, 
              MPI_CHAR, get_request_from, 0, 
              MPI_COMM_WORLD, &status);
        }

        // Copy over the particle data (this also updates the total number of particles)
        copy_over_recieved_data(recv_buffer, ntot_to_recv);
#endif
      }

    template<class T>
      void MPIParticles<T>::dump_to_file(std::string fileprefix, size_t max_bytesize_buffer){
        std::ios_base::sync_with_stdio(false);
        std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
        auto myfile = std::fstream(filename, std::ios::out | std::ios::binary);

        // If we fail to write give a warning, but continue
        if(!myfile.good()) {
          std::string error = "[MPIParticles::dump_to_file] Failed to save the particle data on task " 
            + std::to_string(FML::ThisTask) + " Filename: " + filename;
          std::cout << error << "\n";
          return;
        }

        T a;
        int bytes_per_particle = a.get_particle_byte_size();
        int ndim = a.get_ndim();

        // Write header data
        myfile.write((char*)&bytes_per_particle,   sizeof(bytes_per_particle));
        myfile.write((char*)&ndim,                 sizeof(ndim));
        myfile.write((char*)&NpartTotal,           sizeof(NpartTotal));
        myfile.write((char*)&NpartLocal_in_use,    sizeof(NpartLocal_in_use));
        myfile.write((char*)x_min_per_task.data(), sizeof(double) * FML::NTasks);
        myfile.write((char*)x_max_per_task.data(), sizeof(double) * FML::NTasks);

        // Allocate a write buffer
        size_t buffer_size = NpartTotal * bytes_per_particle < max_bytesize_buffer 
          ? NpartTotal * bytes_per_particle : max_bytesize_buffer;
        std::vector<char> buffer_data(buffer_size);
        size_t n_per_batch = buffer_size / bytes_per_particle-1;
        assert(n_per_batch > 0);

        // Write in chunks
        size_t nwritten = 0;
        while(nwritten < NpartTotal){
          size_t n_to_write = nwritten + n_per_batch < NpartTotal ? n_per_batch : NpartTotal - nwritten;

          char *buffer = buffer_data.data();
          for(size_t i = 0; i < n_to_write; i++){
            p[nwritten + i].append_to_buffer(&buffer[i * bytes_per_particle]);
          }
          myfile.write((char*)buffer, bytes_per_particle * n_to_write);

          nwritten += n_to_write;
        }
        myfile.close();
      }

    template<class T>
      void MPIParticles<T>::load_from_file(std::string fileprefix, size_t max_bytesize_buffer){
        std::ios_base::sync_with_stdio(false);
        std::string filename = fileprefix + "." + std::to_string(FML::ThisTask);
        auto myfile = std::ifstream(filename, std::ios::binary);

        // If we fail to load a file throw an error
        if(!myfile.good()) {
          std::string error = "[MPIParticles::load_from_file] Failed to read the particles on task " 
            + std::to_string(FML::ThisTask) + " Filename: " + filename;
          assert_mpi(false, error.c_str());
        }

        T a;
        int bytes_per_particle_expected = a.get_particle_byte_size();
        int ndim_expected = a.get_ndim();
        int bytes_per_particle;
        int ndim;

        // Read header data
        myfile.read((char*)&bytes_per_particle,   sizeof(bytes_per_particle));
        myfile.read((char*)&ndim,                 sizeof(ndim));
        assert_mpi(bytes_per_particle == bytes_per_particle_expected, 
            "[MPIParticles::load_from_file] Particle byte size do not match the one in the file");
        assert_mpi(ndim == ndim_expected, 
            "[MPIParticles::load_from_file] Particle dimension do not match the one in the file");
        myfile.read((char*)&NpartTotal,           sizeof(NpartTotal));
        myfile.read((char*)&NpartLocal_in_use,    sizeof(NpartLocal_in_use));
        x_min_per_task.resize(FML::NTasks);
        x_max_per_task.resize(FML::NTasks);
        myfile.read((char*)x_min_per_task.data(), sizeof(double) * FML::NTasks);
        myfile.read((char*)x_max_per_task.data(), sizeof(double) * FML::NTasks);

        // Allocate memory
        p.resize(NpartTotal);

        // Allocate a read buffer
        size_t buffer_size = NpartTotal * bytes_per_particle < max_bytesize_buffer 
          ? NpartTotal * bytes_per_particle : max_bytesize_buffer;
        std::vector<char> buffer_data(buffer_size);
        size_t n_per_batch = buffer_size / bytes_per_particle;
        assert(n_per_batch > 0);

        // Read in chunks
        size_t nread = 0;
        while(nread < NpartTotal){
          size_t n_to_read = nread + n_per_batch < NpartTotal ? n_per_batch : NpartTotal - nread;

          char *buffer = buffer_data.data();
          myfile.read((char*)buffer, bytes_per_particle * n_to_read);
          for(size_t i = 0; i < n_to_read; i++){
            p[nread + i].assign_from_buffer(&buffer[i * bytes_per_particle]);
          }

          nread += n_to_read;
        }
        myfile.close();
      }

  }
}
#endif  
