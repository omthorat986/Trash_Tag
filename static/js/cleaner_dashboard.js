/**
 * Cleaner Dashboard - Tab switching and interactions
 */

document.addEventListener("DOMContentLoaded", () => {
  // ===== TAB SWITCHING =====
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const tabName = button.getAttribute('data-tab');

      // Remove active state from all buttons and tabs
      tabButtons.forEach(btn => {
        btn.classList.remove('active');
        btn.setAttribute('aria-selected', 'false');
      });

      tabContents.forEach(content => {
        content.classList.remove('active');
      });

      // Add active state to clicked button and corresponding tab
      button.classList.add('active');
      button.setAttribute('aria-selected', 'true');

      const activeTab = document.getElementById(tabName);
      if (activeTab) {
        activeTab.classList.add('active');
        // Smooth scroll to top of dashboard
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  });

  // ===== IMAGE PREVIEW ON CLICK =====
  document.querySelectorAll('.thumb.preview').forEach(img => {
    img.addEventListener('click', () => {
      const src = img.src;
      const modal = document.createElement('div');
      modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.85);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        cursor: pointer;
      `;

      const imgElement = document.createElement('img');
      imgElement.src = src;
      imgElement.style.cssText = `
        max-width: 90%;
        max-height: 90%;
        border-radius: 8px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
        object-fit: contain;
      `;

      modal.appendChild(imgElement);
      modal.addEventListener('click', () => modal.remove());
      
      // Close on Escape key
      const closeModal = (e) => {
        if (e.key === 'Escape') {
          modal.remove();
          document.removeEventListener('keydown', closeModal);
        }
      };
      document.addEventListener('keydown', closeModal);

      document.body.appendChild(modal);
    });
  });

  // ===== REAL-TIME UPDATES WITH SOCKET.IO =====
  if (typeof io !== 'undefined') {
    try {
      const socket = io();

      socket.on('connect', () => {
        console.log('✅ Socket connected:', socket.id);
        socket.emit('join_cleaners');
      });

      socket.on('new_cleanup_request', (data) => {
        console.log('📬 New request:', data);
        // Optionally notify user without page reload
        // You can replace this with a toast notification
        setTimeout(() => location.reload(), 1500);
      });

      socket.on('request_completed', (payload) => {
        console.log('✅ Request completed:', payload);
        if (payload && payload.request_id) {
          const row = document.querySelector(`[data-request-id="${payload.request_id}"]`);
          if (row) {
            row.style.opacity = '0.5';
            row.style.transition = 'opacity 0.3s ease';
          }
        }
        setTimeout(() => location.reload(), 1000);
      });

      socket.on('request_claimed', (data) => {
        console.log('🔒 Request claimed:', data);
        setTimeout(() => location.reload(), 800);
      });

      socket.on('error', (error) => {
        console.warn('Socket error:', error);
      });

      socket.on('disconnect', () => {
        console.log('🔌 Socket disconnected');
      });

    } catch (error) {
      console.warn('Socket.io initialization failed:', error);
    }
  }

  // ===== FILE INPUT ENHANCEMENT =====
  document.querySelectorAll('.file-input').forEach(input => {
    input.addEventListener('change', (e) => {
      const fileName = e.target.files[0]?.name || 'Choose file';
      // Optionally update label or provide feedback
      console.log('📁 File selected:', fileName);
    });
  });
});
