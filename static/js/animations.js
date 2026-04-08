// animations.js - GSAP based entrance and 3D interactions
if(typeof gsap !== 'undefined'){
  gsap.registerPlugin(ScrollTrigger);
  window.addEventListener('load', ()=>{
    // Hero entrance
    try{
      const heroTitle = document.querySelector('.hero .hero-title');
      const heroLead = document.querySelector('.hero .hero-lead');
      const heroCta = document.querySelector('.hero .hero-cta');
      gsap.from([heroTitle, heroLead, heroCta], { y: 30, opacity: 0, stagger: 0.12, duration: 0.9, ease: 'power3.out' });
    }catch(e){ console.debug('hero anim err', e); }

    // Slight parallax for background shapes
    document.querySelectorAll('[data-parallax]').forEach(el=>{
      const depth = parseFloat(el.getAttribute('data-parallax')||'0.2');
      gsap.to(el, {
        yPercent: depth * -8,
        ease: 'none',
        scrollTrigger: { trigger: el, scrub: 0.6 }
      });
    });

    // 3D card tilt on mouse/touch. We avoid enabling it on pages that should not float (e.g. detect page).
    function bindTilt(containerSelector, strength, opts){
      opts = Object.assign({ perspective: 900, moveDuration: 0.18, scaleOnHover: 1.02, shadowBoost: '0 22px 60px rgba(2,8,23,0.6)' }, opts || {});
      const el = document.querySelector(containerSelector);
      if(!el) return;
      const rect = ()=>el.getBoundingClientRect();

      const onMove = (clientX, clientY)=>{
        const r = rect();
        const px = (clientX - r.left) / r.width;
        const py = (clientY - r.top) / r.height;
        const rx = (py - 0.5) * strength;
        const ry = (px - 0.5) * -strength;
        gsap.to(el, { rotationX: rx, rotationY: ry, transformPerspective: opts.perspective, transformOrigin: 'center', duration: opts.moveDuration, ease: 'power3.out' });
      };

      const mouseMove = (e)=> onMove(e.clientX, e.clientY);
      const touchMove = (e)=>{ if(e.touches && e.touches[0]) onMove(e.touches[0].clientX, e.touches[0].clientY); };

      el.addEventListener('mousemove', mouseMove);
      el.addEventListener('touchmove', touchMove, { passive: true });

      el.addEventListener('mouseenter', ()=>{
        gsap.to(el, { scale: opts.scaleOnHover, duration: 0.18, ease: 'power1.out' });
        gsap.to(el, { boxShadow: opts.shadowBoost, duration: 0.18 });
      });

      el.addEventListener('mouseleave', ()=>{
        gsap.to(el, { rotationX:0, rotationY:0, scale:1, duration:0.28, ease:'power3.out' });
        gsap.to(el, { boxShadow: '0 14px 40px rgba(2,8,23,0.5)', duration: 0.28 });
      });
    }

    // Only bind tilt on pages that are not the detect page OR if the scene explicitly allows it.
    const isDetectPath = (window.location.pathname||'').indexOf('/detect') === 0;
    const sceneAllows = document.querySelector('.scene') && !document.querySelector('.scene').classList.contains('no-3d');
    if(!isDetectPath && sceneAllows){
      bindTilt('.card-3d', 10, { scaleOnHover: 1.03 });
    }

    // Card entrance as they scroll into view
    gsap.utils.toArray('.card-3d, .feature-card, .use-case-card, .tech-item').forEach(card=>{
      gsap.from(card, { y:40, opacity:0, duration:0.9, ease:'power3.out', scrollTrigger:{ trigger: card, start: 'top 85%' } });
    });

    // Section title entrance
    gsap.utils.toArray('.section-title').forEach(title=>{
      gsap.from(title, { y:30, opacity:0, duration:0.8, ease:'power3.out', scrollTrigger:{ trigger: title, start: 'top 90%' } });
    });

    // Workflow steps staggered entrance
    gsap.from('.workflow-steps .step-item', {
      opacity: 0,
      y: 50,
      stagger: 0.2,
      duration: 1,
      ease: 'power3.out',
      scrollTrigger: {
        trigger: '.workflow-steps',
        start: 'top 80%'
      }
    });

    // CTA intro animation
    try{
      const cta = document.querySelector('.btn-getin');
      if(cta){ gsap.from(cta, { y: -8, opacity:0, duration:0.9, ease: 'bounce.out', delay: 0.5 }); }
    }catch(e){ }

    // Small hover animation for nav links
    document.querySelectorAll('.nav-link').forEach(n=>{
      n.addEventListener('mouseenter', ()=> gsap.to(n, { y: -3, duration: 0.18 }))
      n.addEventListener('mouseleave', ()=> gsap.to(n, { y: 0, duration: 0.18 }))
    });

  });
}
