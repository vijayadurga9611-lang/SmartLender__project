// Smart Lender — Main JavaScript

document.addEventListener('DOMContentLoaded', () => {
  console.log('Smart Lender — AI Powered Loan Prediction System');
  
  // Smooth scroll
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Auto-dismiss alerts after 5 seconds
  const alerts = document.querySelectorAll('.alert');
  alerts.forEach(alert => {
    setTimeout(() => {
      const bsAlert = new bootstrap.Alert(alert);
      bsAlert.close();
    }, 5000);
  });

  // Form input formatting
  const incomeInputs = document.querySelectorAll('input[name="applicant_income"], input[name="coapplicant_income"]');
  incomeInputs.forEach(input => {
    input.addEventListener('blur', function() {
      if (this.value) {
        this.value = parseInt(this.value).toLocaleString('en-IN');
      }
    });
    input.addEventListener('focus', function() {
      this.value = this.value.replace(/,/g, '');
    });
  });

  // Navbar shrink on scroll
  let lastScroll = 0;
  window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.glass-nav');
    const currentScroll = window.pageYOffset;
    if (currentScroll > 80) {
      navbar.style.padding = '0.5rem 0';
    } else {
      navbar.style.padding = '0.75rem 0';
    }
    lastScroll = currentScroll;
  });

  // Add intersection observer for animations
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, observerOptions);

  document.querySelectorAll('.feature-card, .model-card, .step-circle').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'all 0.6s ease';
    observer.observe(el);
  });
});
