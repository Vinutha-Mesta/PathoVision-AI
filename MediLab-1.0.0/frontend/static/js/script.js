// script.js - small UI helpers for Detectr+

document.addEventListener('DOMContentLoaded', function () {
  // Navbar shrink on scroll
  const nav = document.querySelector('.navbar');
  function handleNav() {
    if (window.scrollY > 20) nav.classList.add('navbar-sm');
    else nav.classList.remove('navbar-sm');
  }
  handleNav();
  window.addEventListener('scroll', handleNav);

  // File preview in analyze page
  const fileInput = document.querySelector('input[type="file"]#file');
  if (fileInput) {
    const previewWrap = document.createElement('div');
    previewWrap.className = 'text-center mt-3';
    const img = document.createElement('img');
    img.id = 'preview';
    img.className = 'img-entrance';
    previewWrap.appendChild(img);
    fileInput.parentNode.appendChild(previewWrap);

    fileInput.addEventListener('change', (e) => {
      const f = e.target.files[0];
      if (!f) { img.style.display = 'none'; return; }
      const reader = new FileReader();
      reader.onload = function(ev) {
        img.src = ev.target.result;
        img.style.display = 'block';
      }
      reader.readAsDataURL(f);
    });
  }

  // Basic contact form client-side validation (non-blocking)
  const contactForm = document.querySelector('form.row.g-3');
  if (contactForm) {
    contactForm.addEventListener('submit', function(e){
      // allow default for now; placeholder for AJAX later
      // e.preventDefault();
      // implement friendly UI later if required
    });
  }
});