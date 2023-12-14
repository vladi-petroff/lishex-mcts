#include <stdlib.h>
#include <memory>
#include <cstddef>

class Arena {
public:
    explicit Arena(size_t reserved_MB)
    : m_bytes(static_cast<char*>(malloc(reserved_MB << 20))),
      m_size(0),
      m_capacity(reserved_MB << 20) {
          if (m_bytes == nullptr) {

          }
      }

    ~Arena() {
        free(m_bytes);
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    void *current() const {
        return m_bytes + m_size;
    }

    size_t size() const {
        return m_size;
    }

    bool has_space(size_t requested) {
        return m_size + requested < m_capacity;
    }

    /*
    void *allocate(size_t requested) {
        size_t alignment = alignof(std::max_align_t);
        requested = (requested + alignment - 1) & ~(alignment - 1);

        if (m_size + requested > m_capacity) {
            return NULL;
        }
        void *result = current();
        m_size += requested;
        return result;
    }
    */
    void *allocate(size_t requested) {
        size_t alignment = alignof(std::max_align_t);

        // Space available for allocation
        size_t space = m_capacity - m_size;

        void* aligned_ptr = m_bytes + m_size;
        if (std::align(alignment, requested, aligned_ptr, space)) {
            if (space < requested) {
                return nullptr;
            }

            m_size += (static_cast<char*>(aligned_ptr) - (m_bytes + m_size)) + requested;
            return aligned_ptr;
        }

        return nullptr;
    }

    void reset() {
        m_size = 0;
    }

private:
    char *m_bytes;
    size_t m_size;
    size_t m_capacity;
};