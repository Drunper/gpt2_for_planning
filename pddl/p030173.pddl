(define (problem logistics-test)
(:domain logistics)
(:objects
	apn3 apn8 - airplane
	cit3 cit6 - city
	apt8 apt4 - airport
	tru2 tru5 - truck
	obj23 obj99 obj21 obj44 obj11 obj00 - package
	pos77 pos33 pos44 pos23 pos12 pos55 - location
)
(:init
	(at apn3 apt4)
	(at apn8 apt8)
	(at obj23 pos77)
	(at obj99 pos77)
	(at obj21 pos12)
	(at obj44 pos55)
	(at obj11 pos23)
	(at obj00 pos55)
	(in-city apt8 cit6)
	(in-city apt4 cit3)
	(in-city pos77 cit6)
	(in-city pos33 cit3)
	(in-city pos44 cit3)
	(in-city pos23 cit3)
	(in-city pos12 cit6)
	(in-city pos55 cit3)
	(at tru5 pos12)
	(at tru2 pos55)
)
(:goal
	(and
		(at obj23 pos77)
		(at obj99 pos55)
))
)
